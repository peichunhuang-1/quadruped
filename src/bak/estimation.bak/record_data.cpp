#include <iostream>
#include "csv_reader.hpp"
#include "ObservationModel.hpp"
#include "Optimizer.hpp"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "sensor.pb.h"
#include "tool.h"

using encoder = std::pair<std::pair<double, double>, std::pair<double, double> >;

std::mutex mutex_;
motor_msg::MotorStamped motor_data;
void motor_data_cb(motor_msg::MotorStamped msg) {
    mutex_.lock();
    motor_data = msg;
    mutex_.unlock();
}
sensor_msg::IMU imu_data;
void imu_data_cb(sensor_msg::IMU msg) {
    mutex_.lock();
    imu_data = msg;
    mutex_.unlock();
}
sensor_msg::Lidar lidar_data;
void lidar_data_cb(sensor_msg::Lidar msg) {
    mutex_.lock();
    lidar_data = msg;
    mutex_.unlock();
}

encoder phiRL_2_thetabeta(double R, double L, double Rd, double Ld) {
    std::complex<double> r = std::polar(1., R + 17.0 * M_PI / 180.0); 
    std::complex<double> l = std::polar(1., L - 17.0 * M_PI / 180.0); 
    double theta = std::arg(r / l);
    if (theta < 0) {
        theta += 2. * M_PI;
    }
    theta = theta * 0.5;
    double beta = std::arg(l) + theta;
    double beta_d = (Rd + Ld) * 0.5 ;
    double theta_d = (Rd - Ld) * 0.5 ;
    return encoder(std::pair<double, double>(theta, beta), std::pair<double, double>(theta_d, beta_d));
}

int main(int argc, char* argv[]) {
    core::NodeHandler nh;
    core::Subscriber<motor_msg::MotorStamped> &motor_sub = nh.subscribe<motor_msg::MotorStamped>("motor/state", 1000, motor_data_cb);
    core::Subscriber<sensor_msg::IMU> &imu_sub = nh.subscribe<sensor_msg::IMU>("imu", 1000, imu_data_cb);
    core::Subscriber<sensor_msg::Lidar> &dist_sub = nh.subscribe<sensor_msg::Lidar>("lidar", 1000, lidar_data_cb);
    core::Rate rate(1000);
    std::ofstream file;
    std::string saved_file_name = std::string("save");
    file.open(saved_file_name + ".csv");
    file << "Cx" << "," << "Cy" << "," << "Cz" << "," << 
    "v.x" << "," << "v.y" << "," << "v.z" << "," << 
    "a.x" << "," << "a.y" << "," << "a.z" << "," << 
    "w.x" << "," << "w.y" << "," << "w.z" << "," << 
    "q.w" << "," << "q.x" << "," << "q.y" << "," << "q.z" << "," << 
    "lf.theta" << "," << "lf.beta" << "," << "lf.theta_d" << "," << "lf.beta_d" << "," << 
    "rf.theta" << "," << "rf.beta" << "," << "rf.theta_d" << "," << "rf.beta_d" << "," << 
    "rh.theta" << "," << "rh.beta" << "," << "rh.theta_d" << "," << "rh.beta_d" << "," << 
    "lh.theta" << "," << "lh.beta" << "," << "lh.theta_d" << "," << "lh.beta_d" << "," << 
    "lf.contact" << "," << "rf.contact" << "," << "rh.contact" << "," << "lh.contact" <<
    "lf.dst" << "," << "rf.dst" << "," << "rh.dst" << "," << "lh.dst" << "\n";

    while (1) {
        core::spinOnce();
        bool update = false;
        mutex_.lock();
        if (motor_data.motors().size() == 8) {
            encoder lf = phiRL_2_thetabeta(motor_data.motors(0).angle(), motor_data.motors(1).angle(), motor_data.motors(0).twist(), motor_data.motors(1).twist());
            encoder rf = phiRL_2_thetabeta(motor_data.motors(2).angle(), motor_data.motors(3).angle(), motor_data.motors(2).twist(), motor_data.motors(3).twist());
            encoder rh = phiRL_2_thetabeta(motor_data.motors(4).angle(), motor_data.motors(5).angle(), motor_data.motors(4).twist(), motor_data.motors(5).twist());
            encoder lh = phiRL_2_thetabeta(motor_data.motors(6).angle(), motor_data.motors(7).angle(), motor_data.motors(6).twist(), motor_data.motors(7).twist());
            
            ENCODER_DATA elf = {
             lf.first.second, // beta
             lf.first.first, // theta
             lf.second.second, // beta
             lf.second.first, // theta
            };
            ENCODER_DATA erf = {
             rf.first.second, // beta
             rf.first.first, // theta
             rf.second.second, // beta
             rf.second.first, // theta
            };
            ENCODER_DATA erh = {
             rh.first.second, // beta
             rh.first.first, // theta
             rh.second.second, // beta
             rh.second.first, // theta
            };
            ENCODER_DATA elh = {
             lh.first.second, // beta
             lh.first.first, // theta
             lh.second.second, // beta
             lh.second.first, // theta
            };
            if (lidar_data.dist().size() == 4) {
                double alpha_l = atan2(lidar_data.dist(0) - lidar_data.dist(3) , 0.4);
                double alpha_r = atan2(lidar_data.dist(1) - lidar_data.dist(2) , 0.4);
                if (! lidar_data.header().seq() % 50) update = true;
                DST_DATA dlf = {
                    lidar_data.dist(0),
                    alpha_l
                };
                DST_DATA drf = {
                    lidar_data.dist(1),
                    alpha_r
                };
                DST_DATA drh = {
                    lidar_data.dist(2),
                    alpha_r
                };
                DST_DATA dlh = {
                    lidar_data.dist(3),
                    alpha_l
                };
                IMU_DATA imu = {
                    Eigen::Vector3d(imu_data.acceleration().x(), imu_data.acceleration().y(), imu_data.acceleration().z()),
                    Eigen::Vector3d(imu_data.twist().x(), imu_data.twist().y(), imu_data.twist().z()),
                    Eigen::Vector4d(imu_data.orientation().x(), imu_data.orientation().y(), imu_data.orientation().z(), imu_data.orientation().w())
                };

            }
        }
        mutex_.unlock();
        rate.sleep();
    }
    return 0;
}