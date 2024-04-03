#include "KF.hpp"

namespace estimation_model {
    KF::KF(int j, double t) : dt(t) {
        A = Eigen::MatrixXd::Identity(6 * j, 6 * j);
        for (int i = j; i < 2 * j; i ++) {
            A.block<3, 3>((i - j) * 3, i * 3) = -dt * Eigen::Matrix3d::Identity();
        }
        I = Eigen::MatrixXd::Identity(6 * j, 6 * j);
        C = Eigen::MatrixXd::Constant(12 + 6 * (j - 1), j * 6, 0);
        Eigen::Matrix3d dtI = Eigen::Matrix3d::Identity() * dt;
        for (int i = 0; i < j; i++) {
            C.block<3, 3>(0, 3 * i) = dtI ;
            C.block<3, 3>(3, 3 * i) = dtI ;
            C.block<3, 3>(6, 3 * i) = dtI ;
            C.block<3, 3>(9, 3 * i) = dtI ;
            C.block<3, 3>(0, 3 * (i + j)) = -dtI * dt * 0.5 ;
            C.block<3, 3>(3, 3 * (i + j)) = -dtI * dt * 0.5 ;
            C.block<3, 3>(6, 3 * (i + j)) = -dtI * dt * 0.5 ;
            C.block<3, 3>(9, 3 * (i + j)) = -dtI * dt * 0.5 ;
        }
        for (int i = 0; i < j - 1; i ++) {
            C.block<3, 3>(i * 3 + 12, i * 3) = -Eigen::Matrix3d::Identity();
            C.block<3, 3>(i * 3 + 12, i * 3 + 3) = Eigen::Matrix3d::Identity();
            C.block<3, 3>(i * 3 + 12, (i + j) * 3) = -Eigen::Matrix3d::Identity();
            C.block<3, 3>((i + j - 1) * 3 + 12, (i + j) * 3) = -Eigen::Matrix3d::Identity();
            C.block<3, 3>((i + j - 1) * 3 + 12, (i + j) * 3 + 3) = Eigen::Matrix3d::Identity();
        }
        double epsilon = 1e-8;
        P = epsilon * Eigen::MatrixXd::Identity(6 * j, 6 * j);
        R.resize(6 * j, 6 * j) ;
        K.resize(6 * j, 12 + 6 * (j - 1)) ;
        Q.resize(12 + 6 * (j - 1), 12 + 6 * (j - 1)) ;
        x.resize(6 * j) ;
    }
    void KF::init(Eigen::VectorXd x_init) {
        x = x_init;
    }
    void KF::predict(Eigen::VectorXd u, Eigen::MatrixXd noise) {
        R = noise;
        x = A * x + u;
        P = A * P * A.transpose() + R;
    }
    void KF::valid(Eigen::VectorXd z, Eigen::MatrixXd noise) {
        Q = noise;
        K = P * C.transpose() * (C * P * C.transpose() + Q).inverse();
        x = x + K * (z - C * x);
        P = (I - K * C) * P;
    }
    Eigen::VectorXd KF::state() {return x;}
    Eigen::MatrixXd KF::predicted_observation_cov() {return C * P * C.transpose();}
    Eigen::VectorXd KF::measurement() {return C * x;}
    U::U(int size, Eigen::Vector3d a_init, Eigen::Matrix3d R_init) :n(size) {
        for (int i = 0; i < size; i ++) {
            rot.push_back(R_init) ;
            accel.push_back(a_init) ;
        }
    }
    Eigen::MatrixXd U::noise() {
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n * 6, n * 6);
        for (int i = 0; i < n * 3; i ++) R(i, i) = 1e-4 ;
        for (int i = n * 3; i < n * 6; i ++) R(i, i) = 1e-4 ;
        return R;
    }
    Eigen::VectorXd U::u(double dt) {
        Eigen::VectorXd u = Eigen::VectorXd::Zero(n * 6);
        for (int i = 0; i < n; i ++) {
            Eigen::Vector3d a =  dt * rot[i] * accel[i];
            u.segment(i * 3, 3) = a;
        }
        return u;
    }
    Eigen::VectorXd U::compensate(double dt) {
        Eigen::Vector3d sum = Eigen::Vector3d(0, 0, 0);
        for (int i = 0; i < n; i ++) {
            Eigen::Vector3d a = 0.5 * dt * dt * rot[i] * accel[i];
            sum += a;
        }
        Eigen::VectorXd compensate; 
        compensate.resize(12 + 6 * (n - 1));
        for (int i = 0; i < 4; i++)
            compensate.segment(i * 3, 3) = sum;
        for (int i = 12; i < 12 + 6 * (n - 1); i++) 
            compensate(i) = 0;
        return compensate;
    }

    Eigen::MatrixXd U::StateMatrix(double dt) {
        Eigen::MatrixXd A_ = Eigen::MatrixXd::Identity(6 * n, 6 * n);
        for (int i = n; i < 2 * n; i ++) {
            A_.block<3, 3>((i - n) * 3, i * 3) = -dt * rot[i - n];
        }
        return A_;
    }

    Eigen::MatrixXd U::ObservationMatrix(double dt) {
        Eigen::MatrixXd C_ = Eigen::MatrixXd::Zero(6 * n - 6, 6 * n);
        for (int i = 0; i < n - 1; i ++) {
            C_.block<3, 3>(i * 3, i * 3) = -Eigen::Matrix3d::Identity();
            C_.block<3, 3>(i * 3, i * 3 + 3) = Eigen::Matrix3d::Identity();
            C_.block<3, 3>(i * 3, (i + n) * 3) = dt * rot[i+1];
            C_.block<3, 3>((i + n - 1) * 3, (i + n) * 3) = -Eigen::Matrix3d::Identity();
            C_.block<3, 3>((i + n - 1) * 3, (i + n) * 3 + 3) = Eigen::Matrix3d::Identity();
        }
        return C_;
    }
    Eigen::VectorXd U::ObservationVector(double dt) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(6 * n - 6);
        for (int i = 1; i < n; i ++) {
            Eigen::Vector3d a = dt * rot[i] * accel[i];
            v.segment((i - 1) * 3, 3) = a;
        }
        return v;
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
    Eigen::Matrix3d Z::noise() {
        Eigen::Matrix3d Q;
        Q << 1e-6, 0, 0, 
            0, 1e-6, 0,
            0, 0, 1e-6;
        return Q;
    }

    Eigen::MatrixXd Z::concate(Eigen::Matrix3d Q1, Eigen::Matrix3d Q2, Eigen::Matrix3d Q3, Eigen::Matrix3d Q4, Eigen::Matrix3d Q5) {
        Eigen::MatrixXd Q_ = Eigen::MatrixXd::Zero(12 + 6 * (n - 2), 12 + 6 * (n - 2));
        Q_.block<3, 3>(0, 0) = Q1 ;
        Q_.block<3, 3>(3, 3) = Q2 ;
        Q_.block<3, 3>(6, 6) = Q3 ;
        Q_.block<3, 3>(9, 9) = Q4 ;
        for (int i = 12; i < 6 * (n - 2); i+=3) {
            Q_.block<3, 3> (i, i) = Q5;
        }
        return Q_;
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