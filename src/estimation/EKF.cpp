#include "EKF.hpp"

namespace estimation_model {
    EKF::EKF(int j, double t) : dt(t) {
        A = Eigen::MatrixXd::Identity(j, j);
        C = Eigen::MatrixXd::Constant(4, j, t);
        double epsilon1 = 0.0;
        double epsilon2 = 1e-8;
        P = Eigen::MatrixXd::Constant(j, j, epsilon1) + epsilon2 * Eigen::MatrixXd::Identity(j, j);
        R.resize(j, j) ;
        K.resize(j, 4) ;
        Q.resize(4, 4) ;
        x.resize(j, 3) ;
    }
    void EKF::init(Eigen::MatrixXd x_init) {
        x = x_init;
    }
    void EKF::predict(Eigen::MatrixXd u, Eigen::MatrixXd noise) {
        R = noise;
        x = A * x + u;
        P = P + R;
    }
    void EKF::valid(Eigen::MatrixXd z, Eigen::MatrixXd noise) {
        Q = noise;
        K = P * C.transpose() * (C * P * C.transpose() + Q).inverse();
        x = x + K * (z - C * x);
        P = (A - K * C) * P; // A 剛好是identity
    }
    Eigen::MatrixXd EKF::predicted_m() {
        return C * x;
    }
    Eigen::MatrixXd EKF::state() {return x;}

    U::U(int size, Eigen::Vector3d a_init, Eigen::Matrix3d R_init) :n(size) {
        for (int i = 0; i < size; i ++) {
            rot.push_back(R_init) ;
            accel.push_back(a_init) ;
        }
    }
    Eigen::MatrixXd U::noise() {
        Eigen::MatrixXd R;
        R.resize(n, n);
        for (int i = 0; i < n; i ++) R(i, i) = 1e-6 ;
        return R;
    }
    Eigen::MatrixXd U::u(double dt) {
        Eigen::MatrixXd u;
        u.resize(n, 3) ;
        for (int i = 0; i < n; i ++) {
            u.row(i) = dt * rot[i] * accel[i];
        }
        return u;
    }
    void U::push_data(Eigen::Vector3d a, Eigen::Matrix3d R) {
        rot.push_back(R) ;
        accel.push_back(a) ;
        rot.pop_front() ;
        accel.pop_front() ;
    }

    Z::Z(int size, Eigen::Vector<double, 5> encoder_init, Eigen::Matrix3d R_init, double alpha_init) :n(size) {
        for (int i = 0; i < size; i ++) {
            trajectories.push_back(trajectory{encoder_init(0), encoder_init(1), encoder_init(1) + alpha_init, R_init});
            theta_d.push_back(encoder_init(4));
        }
    }
    double Z::noise() {
        return 1e-6;
    }
    Eigen::Vector3d Z::z(Leg &leg, double dt) {
        ContactMap cm;
        Eigen::Vector3d t = cm.travel(trajectories, leg);
        Eigen::Vector3d c = cm.compensate(trajectories, leg, theta_d, dt);
        trajectory last = trajectories.back();
        trajectory first = trajectories.front();
        RIM last_contact_rim = cm.lookup(std::get<0>(last), std::get<2>(last));
        RIM first_contact_rim = cm.lookup(std::get<0>(first), std::get<2>(first));
        leg.Calculate(std::get<0>(last), 0, 0, std::get<1>(last), 0, 0);
        leg.PointContact(last_contact_rim, std::get<2>(last) - std::get<1>(last));
        Eigen::Vector3d last_point = std::get<3>(last) * leg.contact_point;
        leg.Calculate(std::get<0>(first), 0, 0, std::get<1>(first), 0, 0);
        leg.PointContact(first_contact_rim, std::get<2>(first) - std::get<1>(first));
        Eigen::Vector3d first_point = std::get<3>(first) * leg.contact_point;
        return t + first_point - last_point + c;
    }
    void Z::push_data(Eigen::Vector<double, 5> encoders, Eigen::Matrix3d Rk, double dt, double alpha) {
        trajectory last = trajectories.back();
        double contact_beta = (encoders(2) + encoders(3)) * dt + std::get<2>(last);
        if (alpha != -100) contact_beta = encoders(1) + alpha;
        trajectories.push_back(trajectory{encoders(0), encoders(1), contact_beta, Rk});
        trajectories.pop_front();
        theta_d.push_back(encoders(4));
        theta_d.pop_front();
    } // encoders: theta, beta, beta_d, omega

}