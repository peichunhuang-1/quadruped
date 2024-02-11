#include <fstream>
#include "sensor.pb.h"
#include "NodeHandler.h"

std::mutex mutex_;
sensor_msg::IMU imu_info;
void imu_info_cb(sensor_msg::IMU msg)
{
    mutex_.lock();
    imu_info = msg;
    mutex_.unlock();
}

int main() {
    core::NodeHandler nh;
    core::Rate rate(500);
    core::Subscriber<sensor_msg::IMU> &state_sub = nh.subscribe<sensor_msg::IMU>("imu", 800, imu_info_cb);
    std::ofstream file("imu.csv");
    file << "a.x" << "," << "a.y" << "," << "a.z" << "," << 
            "w.x" << "," << "w.y" << "," << "w.z" << "," << 
            "q.x" << "," << "q.y" << "," << "q.z" << "," << "q.w" << 
            "\n";
    while(1) { 
        core::spinOnce();
        mutex_.lock();
        // std::cout << imu_info.orientation().x() << "\t" << imu_info.orientation().y() << "\t" << imu_info.orientation().z() << "\t" << imu_info.orientation().w() << "\n";
        // std::cout << imu_info.acceleration().x() << "\t" << imu_info.acceleration().y() << "\t" << imu_info.acceleration().z() << "\n";
        file << imu_info.acceleration().x() << ",";
        file << imu_info.acceleration().y() << ",";
        file << imu_info.acceleration().z() << ",";
        file << imu_info.twist().x() << ",";
        file << imu_info.twist().y() << ",";
        file << imu_info.twist().z() << ",";
        file << imu_info.orientation().x() << ",";
        file << imu_info.orientation().y() << ",";
        file << imu_info.orientation().z() << ",";
        file << imu_info.orientation().w() << "\n";
        mutex_.unlock();
        rate.sleep();
    }
    file.close();
}