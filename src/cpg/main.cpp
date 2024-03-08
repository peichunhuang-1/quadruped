#include "Kuramoto.hpp" 
#include "fstream"
#include <unistd.h>
#include "kinematic/Leg.hpp"
#include "robot.pb.h"
#include "sensor.pb.h"
#include "motor.pb.h"
#include "NodeHandler.h"
#include "google/protobuf/text_format.h"
using namespace Kuramoto;
using encoder = std::pair<std::pair<double, double>, std::pair<double, double> >;
double beta_op(double x, double beta_max = M_PI_2 / 3., double beta_min = -M_PI_2 / 3.) {
    return (beta_min - beta_max) * x * 0.5;
}

double leg_length_op(double y, double stance_height = 0.2, double dig_depth = 0.001, double lift_height = 0.05) {
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

robot_msg::State state;
void state_cb(robot_msg::State msg)
{
    mutex_.lock();
    state = msg;
    mutex_.unlock();
}

int main() {
    double alpha_ = 5;
    double beta_ = 50;
    double mu_ = 1;
    double freq = 2. * M_PI;
    double omega_stance_ = 1./ 4. * freq;
    double omega_swing_ = 1 * freq;
    double b_ = 1e10;
    kuramoto_neuron lf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron rf(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron rh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    kuramoto_neuron lh(alpha_, beta_, mu_, omega_stance_, omega_swing_, b_) ;
    double dv_lf = 0;
    double dv_rf = 0;
    double dv_rh = 0;
    double dv_lh = 0;

    double force_lf = 0;
    double force_rf = 0;
    double force_rh = 0;
    double force_lh = 0;

    std::ofstream file("kuramoto.csv");
    int counter = 0;
    LinkLegModel lm(0.01, 0.1);

    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);

    core::NodeHandler nh;
    core::Rate rate(1000);
    core::Subscriber<robot_msg::State> &state_sub = nh.subscribe<robot_msg::State>("robot/state", 500, state_cb);
    core::Subscriber<motor_msg::MotorStamped> &motor_sub = nh.subscribe<motor_msg::MotorStamped>("motor/state", 1000, motor_data_cb);
    core::Publisher<motor_msg::MotorStamped> &motor_pub = nh.advertise<motor_msg::MotorStamped>("motor/command");
    core::Subscriber<sensor_msg::IMU> &imu_sub = nh.subscribe<sensor_msg::IMU>("imu", 1000, imu_data_cb);
    for (;;) {
        mutex_.lock();
        core::spinOnce();
        Eigen::Vector4d y(lf.y, rf.y, rh.y, lh.y);
        Eigen::Vector4d Ky;
        Ky = walk_K() * y;
        
        Eigen::Quaterniond q(imu_data.orientation().w(), imu_data.orientation().x(), imu_data.orientation().y(), imu_data.orientation().z());
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Vector3d v(state.twist().linear().x(), state.twist().linear().y(), state.twist().linear().z());
        v = R.transpose() * v;
        Eigen::Vector3d w(state.twist().angular().x(), state.twist().angular().y(), state.twist().angular().z());

        if (motor_data.motors().size() == 8) {   
            force_lf = motor_data.motors(0).torque();
            force_rf = motor_data.motors(2).torque();
            force_rh = motor_data.motors(4).torque();
            force_lh = motor_data.motors(6).torque();
            force_lf = force_lf > 5? force_lf : 0;
            force_rf = force_rf > 5? force_rf : 0;
            force_rh = force_rh > 5? force_rh : 0;
            force_lh = force_lh > 5? force_lh : 0;

            encoder lf_feedback = phiRL_2_thetabeta(motor_data.motors(0).angle(), motor_data.motors(1).angle(), motor_data.motors(0).twist(), motor_data.motors(1).twist());
            encoder rf_feedback = phiRL_2_thetabeta(motor_data.motors(2).angle(), motor_data.motors(3).angle(), motor_data.motors(2).twist(), motor_data.motors(3).twist());
            encoder rh_feedback = phiRL_2_thetabeta(motor_data.motors(4).angle(), motor_data.motors(5).angle(), motor_data.motors(4).twist(), motor_data.motors(5).twist());
            encoder lh_feedback = phiRL_2_thetabeta(motor_data.motors(6).angle(), motor_data.motors(7).angle(), motor_data.motors(6).twist(), motor_data.motors(7).twist());

            lf_leg.Calculate(lf_feedback.first.first, lf_feedback.second.first, 0, lf_feedback.first.second, lf_feedback.second.second, 0);
            lf_leg.PointContact(G_POINT);
            lf_leg.PointVelocity(v, w, G_POINT, 0, false);
            Eigen::Vector3d lf_p = lf_leg.contact_velocity;

            rf_leg.Calculate(rf_feedback.first.first, rf_feedback.second.first, 0, rf_feedback.first.second, rf_feedback.second.second, 0);
            rf_leg.PointContact(G_POINT);
            rf_leg.PointVelocity(v, w, G_POINT, 0, false);
            Eigen::Vector3d rf_p = rf_leg.contact_velocity;

            rh_leg.Calculate(rh_feedback.first.first, rh_feedback.second.first, 0, rh_feedback.first.second, rh_feedback.second.second, 0);
            rh_leg.PointContact(G_POINT);
            rh_leg.PointVelocity(v, w, G_POINT, 0, false);
            Eigen::Vector3d rh_p = rh_leg.contact_velocity;

            lh_leg.Calculate(lh_feedback.first.first, lh_feedback.second.first, 0, lh_feedback.first.second, lh_feedback.second.second, 0);
            lh_leg.PointContact(G_POINT);
            lh_leg.PointVelocity(v, w, G_POINT, 0, false);
            Eigen::Vector3d lh_p = lh_leg.contact_velocity;

            dv_lf = lf_p(0);
            dv_rf = rf_p(0);
            dv_rh = rh_p(0);
            dv_lh = lh_p(0);

            if (lf.y > 0) {
                if (lf.x < -0.95 && lf.x > -1.05 && force_lf > 40 && lf.y < 0.01) {
                    lf.update(Ky(0), 0, dv_lf, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (lf.x > 0) lf.update(Ky(0), 0, dv_lf, 0.001, TRANSITION_TYPE::FEEDBACK);
                else lf.update(Ky(0), force_lf, dv_lf, 0.001, TRANSITION_TYPE::FAST_FEEDBACK);
            }
            else {
                if (lf.x > 0.95 && lf.x < 1.05 && force_lf < 10 && lf.y > -0.01) {
                    lf.update(Ky(0), 0, dv_lf, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (lf.x < 0) lf.update(Ky(0), 0, dv_lf, 0.001, TRANSITION_TYPE::DEFAULT);
                else lf.update(Ky(0), force_lf, dv_lf, 0.001, TRANSITION_TYPE::FAST);
            }
            if (rf.y > 0) {
                if (rf.x < -0.95 && rf.x > -1.05 && force_rf > 40 && rf.y < 0.01) {
                    rf.update(Ky(1), 0, dv_rf, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (rf.x > 0) rf.update(Ky(1), 0, dv_rf, 0.001, TRANSITION_TYPE::FEEDBACK);
                else rf.update(Ky(1), force_rf, dv_rf, 0.001, TRANSITION_TYPE::FAST);
            }
            else {
                if (rf.x > 0.95 && rf.x < 1.05 && force_rf < 10 && rf.y > -0.01) {
                    rf.update(Ky(1), 0, dv_rf, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (rf.x < 0) rf.update(Ky(1), 0, dv_rf, 0.001, TRANSITION_TYPE::DEFAULT);
                else rf.update(Ky(1), force_rf, dv_rf, 0.001, TRANSITION_TYPE::FAST);
            }
            if (rh.y > 0) {
                if (rh.x < -0.95 && rh.x > -1.05 && force_rh > 40 && rh.y < 0.01) {
                    rh.update(Ky(2), 0, dv_rh, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (rh.x > 0) rh.update(Ky(2), 0, dv_rh, 0.001, TRANSITION_TYPE::FEEDBACK);
                else rh.update(Ky(2), force_rh, dv_rh, 0.001, TRANSITION_TYPE::FAST);
            }
            else {
                if (rh.x > 0.95 && rh.x < 1.05 && force_rh < 10 && rh.y > -0.01) {
                    rh.update(Ky(2), 0, dv_rh, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (rh.x < 0) rh.update(Ky(2), 0, dv_rh, 0.001, TRANSITION_TYPE::DEFAULT);
                else rh.update(Ky(2), force_rh, dv_rh, 0.001, TRANSITION_TYPE::FAST);
            }
            if (lh.y > 0) {
                if (lh.x < -0.95 && lh.x > -1.05 && force_lh > 40 && lh.y < 0.01) {
                    lh.update(Ky(3), 0, dv_lh, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (lh.x > 0) lh.update(Ky(3), 0, dv_lh, 0.001, TRANSITION_TYPE::FEEDBACK);
                else lh.update(Ky(3), force_lh, dv_lh, 0.001, TRANSITION_TYPE::FAST);
            }
            else {
                if (lh.x > 0.95 && lh.x < 1.05 && force_lh < 10 && lh.y > -0.01) {
                    lh.update(Ky(3), 0, dv_lh, 0.001, TRANSITION_TYPE::STOP);
                }
                else if (lh.x < 0) lh.update(Ky(3), 0, dv_lh, 0.001, TRANSITION_TYPE::DEFAULT);
                else lh.update(Ky(3), force_lh, dv_lh, 0.001, TRANSITION_TYPE::FAST);
            }
            // lf.update(Ky(0), 0, dv_lf, 0.001, TRANSITION_TYPE::DEFAULT);
            // rf.update(Ky(1), 0, dv_rf, 0.001, TRANSITION_TYPE::DEFAULT);
            // rh.update(Ky(2), 0, dv_rh, 0.001, TRANSITION_TYPE::DEFAULT);
            // lh.update(Ky(3), 0, dv_lh, 0.001, TRANSITION_TYPE::DEFAULT);
            // file << lf_e.first << "," << lf_e.second << "," << rf_e.first << "," << rf_e.second << "," << rh_e.first << "," << rh_e.second << "," << lh_e.first << "," << lh_e.second << "\n";
            file << lf_p(0) << "," << lf_p(1) << "," << lf_p(2) << "," <<
                    rf_p(0) << "," << rf_p(1) << "," << rf_p(2) << "," <<
                    rh_p(0) << "," << rh_p(1) << "," << rh_p(2) << "," <<
                    lh_p(0) << "," << lh_p(1) << "," << lh_p(2) << "," <<
                    lf.x << "," << lf.y << "," <<
                    rf.x << "," << rf.y << "," <<
                    rh.x << "," << rh.y << "," <<
                    lh.x << "," << lh.y << "\n";
        }
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
        
        counter ++;
        mutex_.unlock();
        motor_pub.publish(motor_data_pub);
        rate.sleep();
        if (counter > 30000) break;
    }
    file.close();
}