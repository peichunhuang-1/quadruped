#ifndef KF_HPP
#define KF_HPP
#include <Eigen/Dense>
#include "ContactMap.hpp"
namespace estimation_model {
    class KF {
        public:
        KF(int j, double t) ;
        void predict(Eigen::VectorXd u, Eigen::MatrixXd noise) ;
        void valid(Eigen::VectorXd z, Eigen::MatrixXd noise) ;
        void init(Eigen::VectorXd x_init) ;
        Eigen::VectorXd state();
        Eigen::VectorXd measurement();
        Eigen::MatrixXd predicted_observation_cov();
        Eigen::MatrixXd A;
        Eigen::MatrixXd C;
        private:
        
        Eigen::MatrixXd P;
        Eigen::MatrixXd K;
        Eigen::VectorXd x;
        Eigen::MatrixXd R;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd I;
        double dt;
    };
    class U {
        public:
            U(int size, Eigen::Vector3d a_init, Eigen::Matrix3d R_init) ;
            Eigen::MatrixXd noise() ;
            Eigen::MatrixXd StateMatrix(double dt) ;
            Eigen::MatrixXd ObservationMatrix(double dt) ;
            Eigen::VectorXd ObservationVector(double dt);
            Eigen::VectorXd u(double dt) ;
            Eigen::VectorXd compensate(double dt) ;
            void push_data(Eigen::Vector3d a, Eigen::Matrix3d R) ;
        private:
            std::deque<Eigen::Vector3d> accel;
            std::deque<Eigen::Matrix3d> rot;
            const int n;
    };

    class Z {
        public:
            Z(int size, Eigen::Vector<double, 5> encoder_init, Eigen::Matrix3d R_init, double alpha_init) ;
            Eigen::Matrix3d noise() ;
            Eigen::MatrixXd concate(Eigen::Matrix3d Q1, Eigen::Matrix3d Q2, Eigen::Matrix3d Q3, Eigen::Matrix3d Q4, Eigen::Matrix3d Q5) ;
            Eigen::Vector3d z(Leg &leg, double dt) ;
            void push_data(Eigen::Vector<double, 5> encoders, Eigen::Matrix3d Rk, double dt, double alpha = -100) ; // encoders: theta, beta, beta_d, omega
        private:
            std::deque<trajectory> trajectories;
            std::deque<double> theta_d;
            const int n;
    };
}

#endif