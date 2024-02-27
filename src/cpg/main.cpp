#include "Kuramoto.hpp" 
#include "fstream"
#include <unistd.h>
#include "kinematic/Leg.hpp"
#include "robot.pb.h"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "google/protobuf/text_format.h"
using namespace Kuramoto;

double beta_op(double x, double beta_max = M_PI_2 / 3., double beta_min = -M_PI_2 / 3.) {
    return (beta_min - beta_max) * x * 0.5;
}

double leg_length_op(double y, double stance_height = 0.2, double dig_depth = 0.0001, double lift_height = 0.05) {
    double r;
    if (y > 0) {
        r = stance_height + dig_depth * y;
    }
    else {
        r = (stance_height + lift_height * y);
    }
    return r;
}

std::pair<double, double> theta_beta_2_phiRL(std::pair<double, double> theta_beta) {
    std::pair<double, double> phi_rl = std::pair<double, double> (theta_beta.second + theta_beta.first - 0.296706, theta_beta.second - theta_beta.first + 0.296706);
    return phi_rl;
}

std::mutex mutex_;
robot_msg::State state;
void state_cb(robot_msg::State msg)
{
    mutex_.lock();
    state = msg;
    mutex_.unlock();
}

int main() {
    double alpha_ = 10;
    double beta_ = 10;
    double mu_ = 1;
    double freq = 2. * M_PI;
    double omega_stance_ = 1./ 4. * freq;
    double omega_swing_ = 1 * freq;
    double b_ = 1e10;
    kuramoto_neuron lf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron rf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron rh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron lh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    double dv = 0;
    std::ofstream file("kuramoto.csv");
    int counter = 0;
    LinkLegModel lm(0.01, 0.1);
    Leg leg(Eigen::Vector3d(0, 0, 0), 0.1, 0.01);
    core::NodeHandler nh;
    core::Rate rate(1000);
    core::Subscriber<robot_msg::State> &state_sub = nh.subscribe<robot_msg::State>("robot/state", 500, state_cb);
    core::Publisher<motor_msg::MotorStamped> &motor_pub = nh.advertise<motor_msg::MotorStamped>("motor/command");
    for (;;) {
        Eigen::Vector4d y(lf.y, rf.y, rh.y, lh.y);
        Eigen::Vector4d Ky;
        Ky = walk_K() * y;
        // if (counter < 10000) Ky = walk_K() * y;
        // else {
        //     Ky = trot_K() * y;
        //     omega_stance_ = 1./ 1.5 * freq;
        //     lf.change_param(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
        //     rf.change_param(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
        //     rh.change_param(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
        //     lh.change_param(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_);
        // }
        if (dv < 0.1) dv += 0.0;
        if (lf.y > 0) lf.update(Ky(0), 0, dv, 0.001, TRANSITION_TYPE::FEEDBACK);
        else lf.update(Ky(0), 0, dv, 0.001, TRANSITION_TYPE::DEFAULT);
        if (rf.y > 0) rf.update(Ky(1), 0, dv, 0.001, TRANSITION_TYPE::FEEDBACK);
        else rf.update(Ky(1), 0, dv, 0.001, TRANSITION_TYPE::DEFAULT);
        if (rh.y > 0) rh.update(Ky(2), 0, dv, 0.001, TRANSITION_TYPE::FEEDBACK);
        else rh.update(Ky(2), 0, dv, 0.001, TRANSITION_TYPE::DEFAULT);
        if (lh.y > 0) lh.update(Ky(3), 0, dv, 0.001, TRANSITION_TYPE::FEEDBACK);
        else lh.update(Ky(3), 0, dv, 0.001, TRANSITION_TYPE::DEFAULT);
        double beta_lf = beta_op(lf.x);
        double beta_rf = beta_op(rf.x);
        double beta_rh = beta_op(rh.x);
        double beta_lh = beta_op(lh.x);

        double leg_length_lf = lm.inverse(leg_length_op(lf.y) / cos(beta_lf), G_POINT);
        double leg_length_rf = lm.inverse(leg_length_op(rf.y) / cos(beta_rf), G_POINT);
        double leg_length_rh = lm.inverse(leg_length_op(rh.y) / cos(beta_rh), G_POINT);
        double leg_length_lh = lm.inverse(leg_length_op(lh.y) / cos(beta_lh), G_POINT);

        std::pair<double, double> lf_e = theta_beta_2_phiRL(std::pair<double, double>{leg_length_lf, beta_lf});
        std::pair<double, double> rf_e = theta_beta_2_phiRL(std::pair<double, double>{leg_length_rf, beta_rf});
        std::pair<double, double> rh_e = theta_beta_2_phiRL(std::pair<double, double>{leg_length_rh, beta_rh});
        std::pair<double, double> lh_e = theta_beta_2_phiRL(std::pair<double, double>{leg_length_lh, beta_lh});

        leg.Calculate(leg_length_lf, 0, 0, beta_lf, 0, 0);
        leg.PointContact(G_POINT);
        Eigen::Vector3d lf_p = leg.contact_point;

        leg.Calculate(leg_length_rf, 0, 0, beta_rf, 0, 0);
        leg.PointContact(G_POINT);
        Eigen::Vector3d rf_p = leg.contact_point;

        leg.Calculate(leg_length_rh, 0, 0, beta_rh, 0, 0);
        leg.PointContact(G_POINT);
        Eigen::Vector3d rh_p = leg.contact_point;

        leg.Calculate(leg_length_lh, 0, 0, beta_lh, 0, 0);
        leg.PointContact(G_POINT);
        Eigen::Vector3d lh_p = leg.contact_point;

        
        file << lf_e.first << "," << lf_e.second << "," << rf_e.first << "," << rf_e.second << "," << rh_e.first << "," << rh_e.second << "," << lh_e.first << "," << lh_e.second << "\n";
        motor_msg::MotorStamped motor_data;
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

        motor_data.add_motors()->CopyFrom(lfr);
        motor_data.add_motors()->CopyFrom(lfl);
        motor_data.add_motors()->CopyFrom(rfr);
        motor_data.add_motors()->CopyFrom(rfl);
        motor_data.add_motors()->CopyFrom(rhr);
        motor_data.add_motors()->CopyFrom(rhl);
        motor_data.add_motors()->CopyFrom(lhr);
        motor_data.add_motors()->CopyFrom(lhl);

        motor_pub.publish(motor_data);
        counter ++;
        rate.sleep();
        if (counter > 30000) break;
    }
    file.close();
}