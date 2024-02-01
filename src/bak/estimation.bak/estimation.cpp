#include <iostream>
#include "csv_reader.hpp"
#include "ObservationModel.hpp"
#include "Optimizer.hpp"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "sensor.pb.h"

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
    LegVelocityEstimation lf_leg(Eigen::Vector3d(0.2, 0.15, 0), Eigen::Vector3d(0.2 , 0.08, 0), 0.1, 0.01, 0.001);
    LegVelocityEstimation rf_leg(Eigen::Vector3d(0.2, -0.15, 0), Eigen::Vector3d(0.2 ,-0.08, 0), 0.1, 0.01, 0.001);
    LegVelocityEstimation rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), Eigen::Vector3d(-0.2 ,-0.08, 0), 0.1, 0.01, 0.001);
    LegVelocityEstimation lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), Eigen::Vector3d(-0.2 ,0.08, 0), 0.1, 0.01, 0.001);
    BodyEstimation fopt_p(&lf_leg, &rf_leg, &rh_leg, &lh_leg, Eigen::Vector3d(0, 0, 0));
    fopt_p.current_position = Eigen::Vector3d(0, 0, 0.11);
    nlopt::opt fopt = Optimizer(&fopt_p);
    std::vector<double> u {0, 0, 0, .5, .5, .5, .5};
    double minf = 0;
    ContactMap cm;

    core::NodeHandler nh;
    core::Subscriber<motor_msg::MotorStamped> &motor_sub = nh.subscribe<motor_msg::MotorStamped>("motor/state", 1000, motor_data_cb);
    core::Subscriber<sensor_msg::IMU> &imu_sub = nh.subscribe<sensor_msg::IMU>("imu", 1000, imu_data_cb);
    core::Subscriber<sensor_msg::Lidar> &dist_sub = nh.subscribe<sensor_msg::Lidar>("lidar", 1000, lidar_data_cb);
    core::Rate rate(1000);
    bool init = false;

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

                STATE lf_v, rf_v, rh_v, lh_v;
                if (! init) {
                    update = true;
                    init = true;
                }
                lf_v = lf_leg.calculate(imu, elf, dlf, update);
                rf_v = rf_leg.calculate(imu, erf, drf, update);
                rh_v = rh_leg.calculate(imu, erh, drh, update);
                lh_v = lh_leg.calculate(imu, elh, dlh, update);
                // optimize
                try {
                fopt.optimize(u, minf);
                }catch (const nlopt::roundoff_limited &ex) {
                    std::cerr << "Caught nlopt::roundoff_limited exception: " << ex.what() << std::endl;
                }

                fopt_p.prediction();
                fopt_p.update_weight(update);
                std::cout << "sequence : \t" << lidar_data.header().seq() << "\n";
                std::cout << "position : \t" << fopt_p.current_position(0) << "\t" << fopt_p.current_position(1) << "\t" << fopt_p.current_position(2) << "\n";
                std::cout << "velocity : \t" << u[0] << "\t" << u[1] << "\t" << u[2] << "\n";
                std::cout << "imu accel : \t" << imu.a.transpose() << "\n";
                std::cout << "imu twist : \t" << imu.w.transpose() << "\n";
                std::cout << "imu orient : \t" << imu.q.transpose() << "\n";
                std::cout << "dist : \t" << lidar_data.dist(0) << "\t" << lidar_data.dist(1) << "\t" << lidar_data.dist(2) << "\t" << lidar_data.dist(3) << "\n";
                std::cout << "lf : \t" << elf.theta << "\t" << elf.beta << "\t" << elf.theta_d << "\t" << elf.beta_d << "\n";
                std::cout << "rf : \t" << erf.theta << "\t" << erf.beta << "\t" << erf.theta_d << "\t" << erf.beta_d << "\n";
                std::cout << "rh : \t" << erh.theta << "\t" << erh.beta << "\t" << erh.theta_d << "\t" << erh.beta_d << "\n";
                std::cout << "lh : \t" << elh.theta << "\t" << elh.beta << "\t" << elh.theta_d << "\t" << elh.beta_d << "\n";
                std::cout << "==============================================================\n";
            }
        }
        mutex_.unlock();
        rate.sleep();
    }
    return 0;
}