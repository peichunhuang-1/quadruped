#include "KF.hpp"
#include <iostream>
#include "csv_reader.hpp"
#include <random>
#include "motor.pb.h"
#include "robot.pb.h"
#include "NodeHandler.h"
#include "sensor.pb.h"

using namespace estimation_model;

template<size_t n>
Eigen::Vector<double, n> random_vector() {
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::uniform_real_distribution<double> dis(-1, 1);
    Eigen::Vector<double, n> V = Eigen::Vector<double, n>().NullaryExpr([&](){return dis(gen);});
    return V;
}

double random_number() {
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::uniform_real_distribution<double> dis(-1, 1);
    return dis(gen);
}

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
    int j = 10;
    double dt = 0.001;
    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);

    Eigen::Vector3d a(0, 0, 0);
    Eigen::Quaterniond q(1, 0, 0, 0);
    Eigen::Vector<double, 5> contact_vector_lf(17.0 / 180.0 * M_PI, 0, 0, 0, 0);
    Eigen::Vector<double, 5> contact_vector_rf(17.0 / 180.0 * M_PI, 0, 0, 0, 0);
    Eigen::Vector<double, 5> contact_vector_rh(17.0 / 180.0 * M_PI, 0, 0, 0, 0);
    Eigen::Vector<double, 5> contact_vector_lh(17.0 / 180.0 * M_PI, 0, 0, 0, 0);
    Eigen::Vector3d v_init(0, 0, 0) ;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(6 * j);
    Eigen::Matrix3d R = q.toRotationMatrix();
    x.resize(6 * j);
    for (int i = 0; i < j; i++) {
        x(i * 3) = v_init(0);
        x(i * 3 + 1) = v_init(1);
        x(i * 3 + 2) = v_init(2);
        x((i + j) * 3) = 1e-3;
        x((i + j) * 3 + 1) = 1e-3;
        x((i + j) * 3 + 2) = 1e-3;
    }

    KF filter(j, dt) ;
    filter.init(x);
    U input(j, a, R) ;
    Z observed_lf(j+1, contact_vector_lf, R, 0);
    Z observed_rf(j+1, contact_vector_rf, R, 0);
    Z observed_rh(j+1, contact_vector_rh, R, 0);
    Z observed_lh(j+1, contact_vector_lh, R, 0);

    core::NodeHandler nh;
    core::Subscriber<motor_msg::MotorStamped> &motor_sub = nh.subscribe<motor_msg::MotorStamped>("motor/state", 1000, motor_data_cb);
    core::Subscriber<sensor_msg::IMU> &imu_sub = nh.subscribe<sensor_msg::IMU>("imu", 1000, imu_data_cb);
    core::Subscriber<sensor_msg::Lidar> &dist_sub = nh.subscribe<sensor_msg::Lidar>("lidar", 20, lidar_data_cb);
    core::Publisher<robot_msg::State> &state_pub = nh.advertise<robot_msg::State>("robot/state");
    core::Rate rate(1000);
    int i = 0;
    std::deque<Eigen::Vector3d> a_2; a_2.push_back(Eigen::Vector3d(0, 0, 0)); a_2.push_back(Eigen::Vector3d(0, 0, 0));
    std::deque<Eigen::Vector4d> q_2; q_2.push_back(Eigen::Vector4d(0, 0, 0, 1)); q_2.push_back(Eigen::Vector4d(0, 0, 0, 1));
    std::ofstream file("velocity_estimation.csv");
    for (;;) {
        core::spinOnce();
        mutex_.lock();
        robot_msg::State s;
        if (motor_data.motors().size() == 8 && lidar_data.dist().size() == 4) {
            encoder lf = phiRL_2_thetabeta(motor_data.motors(0).angle(), motor_data.motors(1).angle(), motor_data.motors(0).twist(), motor_data.motors(1).twist());
            encoder rf = phiRL_2_thetabeta(motor_data.motors(2).angle(), motor_data.motors(3).angle(), motor_data.motors(2).twist(), motor_data.motors(3).twist());
            encoder rh = phiRL_2_thetabeta(motor_data.motors(4).angle(), motor_data.motors(5).angle(), motor_data.motors(4).twist(), motor_data.motors(5).twist());
            encoder lh = phiRL_2_thetabeta(motor_data.motors(6).angle(), motor_data.motors(7).angle(), motor_data.motors(6).twist(), motor_data.motors(7).twist());
            
            Eigen::Vector3d a = a_2.front(); a += Eigen::Vector3d(1e-3, 1e-3, 1e-3); a_2.pop_front(); a_2.push_back(Eigen::Vector3d(imu_data.acceleration().x(), imu_data.acceleration().y(), imu_data.acceleration().z()));
            Eigen::Quaterniond qk_2 = Eigen::Quaterniond(q_2.front()); q_2.pop_front(); q_2.push_back(Eigen::Vector4d(imu_data.orientation().x(), imu_data.orientation().y(), imu_data.orientation().z(), imu_data.orientation().w()));
            Eigen::Matrix3d R_k2 = qk_2.toRotationMatrix();
            Eigen::Quaterniond q = Eigen::Quaterniond(q_2.back());
            Eigen::Vector<double, 5> contact_vector_lf(lf.first.first, lf.first.second, lf.second.second, imu_data.twist().y(), lf.second.first);
            Eigen::Vector<double, 5> contact_vector_rf(rf.first.first, rf.first.second, rf.second.second, imu_data.twist().y(), rf.second.first);
            Eigen::Vector<double, 5> contact_vector_rh(rh.first.first, rh.first.second, rh.second.second, imu_data.twist().y(), rh.second.first);
            Eigen::Vector<double, 5> contact_vector_lh(lh.first.first, lh.first.second, lh.second.second, imu_data.twist().y(), lh.second.first);
            Eigen::Vector3d v(0, 0, 0) ;
            Eigen::VectorXd x;
            Eigen::VectorXd z;
            Eigen::MatrixXd A;
            Eigen::MatrixXd Q;
            z.resize(12);
            Q.resize(12, 12);
            Eigen::Matrix3d R = q.toRotationMatrix();
            std::vector<bool> contact_metrice = {true, true, true, true};

            input.push_data(a, R_k2);
            A = input.StateMatrix(dt);
            double alpha_lf, alpha_rf, alpha_rh, alpha_lh;
            if (i % 4) {
                alpha_lf = -100;
                alpha_rf = -100;
                alpha_rh = -100;
                alpha_lh = -100;
            }
            else {
                alpha_lf = - atan2(lidar_data.dist(0) - lidar_data.dist(3) , 0.4);
                alpha_rf = - atan2(lidar_data.dist(1) - lidar_data.dist(2) , 0.4);
                alpha_rh = - atan2(lidar_data.dist(1) - lidar_data.dist(2) , 0.4);
                alpha_lh = - atan2(lidar_data.dist(0) - lidar_data.dist(3) , 0.4);
            }
            observed_lf.push_data(contact_vector_lf, R, dt, alpha_lf);
            observed_rf.push_data(contact_vector_rf, R, dt, alpha_rf);
            observed_rh.push_data(contact_vector_rh, R, dt, alpha_rh);
            observed_lh.push_data(contact_vector_lh, R, dt, alpha_lh);

            z.segment(0, 3) = observed_lf.z(lf_leg, dt);
            z.segment(3, 3) = observed_rf.z(rf_leg, dt);
            z.segment(6, 3) = observed_rh.z(rh_leg, dt);
            z.segment(9, 3) = observed_lh.z(lh_leg, dt);

            z = z - input.compensate(dt);

            filter.A = A;
            filter.predict(input.u(dt), input.noise());

            Q = observed_lf.concate(observed_lf.noise(), observed_rf.noise(), observed_rh.noise(), observed_lh.noise());
            double min_malahanobis_distance = 1e10;
            int best_observation = 0;
            std::vector<int> kick_out_list;
            for (int k = 0; k < 4; k ++) {
                double malahanobis_distance = (z.segment(k * 3, 3) - filter.measurement().segment(k * 3, 3) ).transpose() * (filter.predicted_observation_cov().block<3, 3>(k * 3, 0)).inverse() * (z.segment(k * 3, 3) - filter.measurement().segment(k * 3, 3));
                if (malahanobis_distance < min_malahanobis_distance) {
                    min_malahanobis_distance = malahanobis_distance;
                    best_observation = k;
                }
                if (malahanobis_distance > 4) {
                    kick_out_list.push_back(k);
                }
            }
            for (auto &iter: kick_out_list) {
                if (iter == best_observation) continue;
                contact_metrice[iter] = false;
                Q.block<3, 3> (iter * 3, iter * 3) = 1e100 * Eigen::Matrix3d::Identity();
            }

            filter.valid(z, Q);
            x = filter.state();
            file << x.segment(3 * j - 3, 3)(0) << "," << x.segment(3 * j - 3, 3)(1) << "," << x.segment(3 * j - 3, 3)(2) <<
            "," << x.segment(6 * j - 3, 3)(0) << "," << x.segment(6 * j - 3, 3)(1) << "," << x.segment(6 * j - 3, 3)(2) << std::endl;
            auto twist = s.mutable_twist();
            
            auto angular = twist->mutable_angular();
            auto linear = twist->mutable_linear();
            
            angular->set_x(imu_data.twist().x());
            angular->set_y(imu_data.twist().y());
            angular->set_z(imu_data.twist().z());

            linear->set_x(x.segment(3 * j - 3, 3)(0));
            linear->set_y(x.segment(3 * j - 3, 3)(1));
            linear->set_z(x.segment(3 * j - 3, 3)(2));
        }
        mutex_.unlock();
        state_pub.publish(s);
        i++;
        rate.sleep();
    }
    return 0;
}