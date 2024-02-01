#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include "robot.pb.h"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "google/protobuf/text_format.h"
#include "kinematic/Leg.hpp"

std::pair<double, double> theta_beta_2_phiRL(std::pair<double, double> theta_beta) {
    std::pair<double, double> phi_rl = std::pair<double, double> (theta_beta.second + theta_beta.first - 0.296706, theta_beta.second - theta_beta.first + 0.296706);
    return phi_rl;
}

std::mutex mutex_;
robot_msg::GaitInfo gait_info;
void gait_info_cb(robot_msg::GaitInfo msg)
{
    mutex_.lock();
    gait_info = msg;
    mutex_.unlock();
}

robot_msg::State state;
void state_cb(robot_msg::State msg)
{
    mutex_.lock();
    state = msg;
    mutex_.unlock();
}

int main()
{
    core::NodeHandler nh;
    core::Rate rate(200);
    core::Subscriber<robot_msg::GaitInfo> &gait_sub = nh.subscribe<robot_msg::GaitInfo>("gait/trajectory", 500, gait_info_cb);
    core::Subscriber<robot_msg::State> &state_sub = nh.subscribe<robot_msg::State>("robot/state", 500, state_cb);
    core::Publisher<motor_msg::MotorStamped> &motor_pub = nh.advertise<motor_msg::MotorStamped>("motor/command");
    std::ofstream file("trajectory.csv");
    std::ofstream file2("trajectory2.csv");
    Leg lf(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.0125);
    Leg rf(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.0125);
    Leg rh(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.0125);
    Leg lh(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.0125);
    while(1) {
        core::spinOnce();
        mutex_.lock();
        if (gait_info.position().pose().size() == 4) {
            Eigen::Vector3d centroid = Eigen::Vector3d(gait_info.centroid().position().x(), gait_info.centroid().position().y(), gait_info.centroid().position().z());
            Eigen::Vector3d lf_contact = Eigen::Vector3d(gait_info.position().pose(0).position().x(), gait_info.position().pose(0).position().y(), gait_info.position().pose(0).position().z());
            Eigen::Vector3d rf_contact = Eigen::Vector3d(gait_info.position().pose(1).position().x(), gait_info.position().pose(1).position().y(), gait_info.position().pose(1).position().z());
            Eigen::Vector3d rh_contact = Eigen::Vector3d(gait_info.position().pose(2).position().x(), gait_info.position().pose(2).position().y(), gait_info.position().pose(2).position().z());
            Eigen::Vector3d lh_contact = Eigen::Vector3d(gait_info.position().pose(3).position().x(), gait_info.position().pose(3).position().y(), gait_info.position().pose(3).position().z());
            auto lf_motors = theta_beta_2_phiRL(lf.Inverse(lf_contact - centroid, G_POINT));
            auto rf_motors = theta_beta_2_phiRL(rf.Inverse(rf_contact - centroid, G_POINT));
            auto rh_motors = theta_beta_2_phiRL(rh.Inverse(rh_contact - centroid, G_POINT));
            auto lh_motors = theta_beta_2_phiRL(lh.Inverse(lh_contact - centroid, G_POINT));
            file2 << centroid(0) << "," << centroid(1) << "," << centroid(2) << "," << 
            lf_contact(0) << "," << lf_contact(1) << "," << lf_contact(2) << "," <<
            rf_contact(0) << "," << rf_contact(1) << "," << rf_contact(2) << "," <<
            rh_contact(0) << "," << rh_contact(1) << "," << rh_contact(2) << "," <<
            lh_contact(0) << "," << lh_contact(1) << "," << lh_contact(2) << "\n";
            // file << (lf_contact - centroid)(0) - 0.2 << "," << (rf_contact - centroid)(0) - 0.2 << "," << (rh_contact - centroid)(0) + 0.2 << "," << (lh_contact - centroid)(0) + 0.2 << "\n";
            motor_msg::MotorStamped motor_data;
            motor_msg::Motor lfr, lfl, rfr, rfl, rhr, rhl, lhr, lhl;
            // std::cout << lf_motors.second << "\n" << lf_motors.first << "\n";
            lfr.set_angle(-lf_motors.second); 
            lfr.set_ki(0);
            lfr.set_kp(90);
            lfr.set_kd(1.75);
            lfl.set_angle(-lf_motors.first);
            lfl.set_ki(0);
            lfl.set_kp(90);
            lfl.set_kd(1.75);
            rfr.set_angle(rf_motors.first); 
            rfr.set_ki(0);
            rfr.set_kp(90);
            rfr.set_kd(1.75);
            rfl.set_angle(rf_motors.second);
            rfl.set_ki(0);
            rfl.set_kp(90);
            rfl.set_kd(1.75);
            rhr.set_angle(rh_motors.first); 
            rhr.set_ki(0);
            rhr.set_kp(90);
            rhr.set_kd(1.75);
            rhl.set_angle(rh_motors.second);
            rhl.set_ki(0);
            rhl.set_kp(90);
            rhl.set_kd(1.75);
            lhr.set_angle(-lh_motors.second); 
            lhr.set_ki(0);
            lhr.set_kp(90);
            lhr.set_kd(1.75);
            lhl.set_angle(-lh_motors.first);
            lhl.set_ki(0);
            lhl.set_kp(90);
            lhl.set_kd(1.75);

            motor_data.add_motors()->CopyFrom(lfr);
            motor_data.add_motors()->CopyFrom(lfl);
            motor_data.add_motors()->CopyFrom(rfr);
            motor_data.add_motors()->CopyFrom(rfl);
            motor_data.add_motors()->CopyFrom(rhr);
            motor_data.add_motors()->CopyFrom(rhl);
            motor_data.add_motors()->CopyFrom(lhr);
            motor_data.add_motors()->CopyFrom(lhl);

            motor_pub.publish(motor_data);

            file << lf_motors.first << "," << lf_motors.second << "," << 
            rf_motors.first << "," << rf_motors.second << "," << 
            rh_motors.first << "," << rh_motors.second << "," << 
            lh_motors.first << "," << lh_motors.second << "," << 
            gait_info.phase().contact(0) << "," << gait_info.phase().contact(1) << "," << gait_info.phase().contact(2) << "," << gait_info.phase().contact(3) << "\n";
        }
        mutex_.unlock();
        rate.sleep();
    }
    return 0;
}