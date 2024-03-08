#include "MyCPGs.hpp"
#include "fstream"
#include <unistd.h>
#include "robot.pb.h"
#include "sensor.pb.h"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "google/protobuf/text_format.h"

using namespace Kuramoto;
using encoder = std::pair<std::pair<double, double>, std::pair<double, double> >;
std::pair<double, double> theta_beta_2_phiRL(std::pair<double, double> theta_beta) {
    std::pair<double, double> phi_rl = std::pair<double, double> (theta_beta.second + theta_beta.first - 0.296706, theta_beta.second - theta_beta.first + 0.296706);
    return phi_rl;
}

int main() {
    double alpha_ = 5;
    double beta_ = 5;
    double mu_ = 1;
    double f = 2. * M_PI / 1.;
    double omega_stance_ = f / 5.;
    double omega_swing_ = f;
    double b_ = 1e10;
    double dt = 0.001;
    kuramoto_neuron lf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
    kuramoto_neuron rf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
    kuramoto_neuron rh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
    kuramoto_neuron lh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);

    std::ofstream file("mycpg.csv");

    core::NodeHandler nh;
    core::Rate rate(1000);
    core::Publisher<motor_msg::MotorStamped> &motor_pub = nh.advertise<motor_msg::MotorStamped>("motor/command");
    int counter = 0;
    for (;;) {
        core::spinOnce();
        Eigen::Vector4d Ky = walk_K() * Eigen::Vector4d(lf.y, rf.y, rh.y, lh.y);
        double Flf, Frf, Frh, Flh;
        double F = 0;
        Flf = lf.y > 0? F : 0;
        Frf = rf.y > 0? F : 0;
        Frh = rh.y > 0? F : 0;
        Flh = lh.y > 0? F : 0;
        lf.update(Ky(0), Flf, dt);
        rf.update(Ky(1), Frf, dt);
        rh.update(Ky(2), Frh, dt);
        lh.update(Ky(3), Flh, dt);

        file << lf.x << "," << lf.y << "," << rf.x << "," << rf.y << "," << rh.x << "," << rh.y << "," << lh.x << "," << lh.y << "\n";

        std::pair<double, double> lf_e, rf_e, rh_e, lh_e;
        lf_e = theta_beta_2_phiRL(lf.output(-M_PI_2 / 3., 0.05, 0.001, lf_leg, dt));
        rf_e = theta_beta_2_phiRL(rf.output(-M_PI_2 / 3., 0.05, 0.001, rf_leg, dt));
        rh_e = theta_beta_2_phiRL(rh.output(-M_PI_2 / 3., 0.05, 0.001, rh_leg, dt));
        lh_e = theta_beta_2_phiRL(lh.output(-M_PI_2 / 3., 0.05, 0.001, lh_leg, dt));
        motor_msg::MotorStamped motor_data_pub;
        motor_msg::Motor lfr, lfl, rfr, rfl, rhr, rhl, lhr, lhl;
        // std::cout << lf_motors.second << "\n" << lf_motors.first << "\n";
        
        lfr.set_angle(-lf_e.second); 
        lfr.set_ki(0);
        lfr.set_kp(90);
        lfr.set_kd(1.75);
        lfl.set_angle(-lf_e.first);
        lfl.set_ki(0);
        lfl.set_kp(90);
        lfl.set_kd(1.75);
        rfr.set_angle(rf_e.first); 
        rfr.set_ki(0);
        rfr.set_kp(90);
        rfr.set_kd(1.75);
        rfl.set_angle(rf_e.second);
        rfl.set_ki(0);
        rfl.set_kp(90);
        rfl.set_kd(1.75);
        rhr.set_angle(rh_e.first); 
        rhr.set_ki(0);
        rhr.set_kp(90);
        rhr.set_kd(1.75);
        rhl.set_angle(rh_e.second);
        rhl.set_ki(0);
        rhl.set_kp(90);
        rhl.set_kd(1.75);
        lhr.set_angle(-lh_e.second); 
        lhr.set_ki(0);
        lhr.set_kp(90);
        lhr.set_kd(1.75);
        lhl.set_angle(-lh_e.first);
        lhl.set_ki(0);
        lhl.set_kp(90);
        lhl.set_kd(1.75);

        motor_data_pub.add_motors()->CopyFrom(lfr);
        motor_data_pub.add_motors()->CopyFrom(lfl);
        motor_data_pub.add_motors()->CopyFrom(rfr);
        motor_data_pub.add_motors()->CopyFrom(rfl);
        motor_data_pub.add_motors()->CopyFrom(rhr);
        motor_data_pub.add_motors()->CopyFrom(rhl);
        motor_data_pub.add_motors()->CopyFrom(lhr);
        motor_data_pub.add_motors()->CopyFrom(lhl);
        
        motor_pub.publish(motor_data_pub);
        rate.sleep();
        counter ++;
    }
    return 0;
}