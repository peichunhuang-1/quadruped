#include "cx5.hpp"
#include "sensor.pb.h"
#include <sys/time.h>
std::shared_ptr<CX5_AHRS> imu;
std::mutex cb_lock;
sensor_msg::SensorMode mode;
void cb(sensor_msg::SensorRequest request, sensor_msg::SensorReply &reply) {
    cb_lock.lock();
    switch (request.mode()) {
        case sensor_msg::REST :
        {
            mode = sensor_msg::REST;
            reply.set_mode(sensor_msg::REST);
            break;
        }
        case sensor_msg::CALIBRATION :
        {
            imu->calibrate(1000); // 1 second averaging
            mode = sensor_msg::SENSOR;
            break;
        }
        case sensor_msg::RESET : 
        {
            mode = sensor_msg::RESET;
            reply.set_mode(sensor_msg::RESET);
            break;
        }
        case sensor_msg::SENSOR :
        {    
            mode = sensor_msg::SENSOR;
            reply.set_mode(sensor_msg::SENSOR);
            break;
        }
    }
    cb_lock.unlock();
}

int main() {
    imu = std::make_shared<CX5_AHRS>("/dev/ttyACM0", 921600, 500, 500);
    core::NodeHandler nh;
    core::Rate rate(500);
    core::Publisher<sensor_msg::IMU> &pub = nh.advertise<sensor_msg::IMU>("imu");
    core::ServiceServer<sensor_msg::SensorRequest, sensor_msg::SensorReply> &srv = nh.serviceServer<sensor_msg::SensorRequest, sensor_msg::SensorReply>("imu", cb);
    std::thread imu_thread = std::thread(
        [&] () {
            ::imu->start();
        }
    );
    sensor_msg::IMU imu_msg;
    imu_msg.mutable_header()->set_frameid("imu_base");
    Eigen::Vector3f acceleration;
    Eigen::Vector3f twist;
    Eigen::Quaternionf orientation;
    int seq = 0;
    while (1) {
        imu->get(acceleration, twist, orientation);
        if (orientation.norm() >= 1e-2) {
            imu->calibrate(1000, true);
            break;
        }
        rate.sleep();
    }
    while (1) {
        cb_lock.lock();
        // if (mode == sensor_msg::SENSOR){
            seq ++;
            imu->get(acceleration, twist, orientation);
            timeval currentTime;
            gettimeofday(&currentTime, nullptr);
            imu_msg.mutable_header()->set_seq(seq);
            imu_msg.mutable_header()->mutable_stamp()->set_sec(currentTime.tv_sec);
            imu_msg.mutable_header()->mutable_stamp()->set_usec(currentTime.tv_usec);
            imu_msg.mutable_acceleration()->set_x(acceleration(0));
            imu_msg.mutable_acceleration()->set_y(acceleration(1));
            imu_msg.mutable_acceleration()->set_z(acceleration(2));
            imu_msg.mutable_twist()->set_x(twist(0));
            imu_msg.mutable_twist()->set_y(twist(1));
            imu_msg.mutable_twist()->set_z(twist(2));
            imu_msg.mutable_orientation()->set_x(orientation.x());
            imu_msg.mutable_orientation()->set_y(orientation.y());
            imu_msg.mutable_orientation()->set_z(orientation.z());
            imu_msg.mutable_orientation()->set_w(orientation.w());
            pub.publish(imu_msg);
        // }
        // std::cout << orientation.x() << "\t" << orientation.y() << "\t" << orientation.z() << "\t" << orientation.w() << "\n";
        cb_lock.unlock();
        rate.sleep();
    }
    return 0;
}