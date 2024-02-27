#ifndef MEASURE_MODEL_HPP
#define MEASURE_MODEL_HPP
#include <Eigen/Dense>
#include "ContactMap.hpp"
namespace estimation_model {
    class EKF {
        public:
        EKF(int j, double t) ;
        void predict(Eigen::MatrixXd u, Eigen::MatrixXd noise) ;
        void valid(Eigen::MatrixXd z, Eigen::MatrixXd noise) ;
        void init(Eigen::MatrixXd x_init) ;
        Eigen::MatrixXd state();
        Eigen::MatrixXd predicted_m();
        private:
        Eigen::MatrixXd A;
        Eigen::MatrixXd P;
        Eigen::MatrixXd C;
        Eigen::MatrixXd K;
        Eigen::MatrixXd x;
        Eigen::MatrixXd R;
        Eigen::MatrixXd Q;
        double dt;
    };
    class U {
        public:
            U(int size, Eigen::Vector3d a_init, Eigen::Matrix3d R_init) ;
            Eigen::MatrixXd noise() ;
            Eigen::MatrixXd u(double dt) ;
            void push_data(Eigen::Vector3d a, Eigen::Matrix3d R) ;
        private:
            std::deque<Eigen::Vector3d> accel;
            std::deque<Eigen::Matrix3d> rot;
            const int n;
    };

    class Z {
        public:
            Z(int size, Eigen::Vector<double, 5> encoder_init, Eigen::Matrix3d R_init, double alpha_init) ;
            double noise() ;
            Eigen::Vector3d z(Leg &leg, double dt) ;
            void push_data(Eigen::Vector<double, 5> encoders, Eigen::Matrix3d Rk, double dt, double alpha = -100) ; // encoders: theta, beta, beta_d, omega
        private:
            std::deque<trajectory> trajectories;
            std::deque<double> theta_d;
            const int n;
    };
}

#endif