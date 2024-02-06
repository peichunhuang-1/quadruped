#ifndef DEFINITION_HPP
#define DEFINITION_HPP
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "ContactMap.hpp"
#include <map>
namespace estimator {
    double relu(double x, double mid_point=0.03, double max=1.) ;
    double gaussianLikelihood(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance);
    class ground {
        public:
            ground() ;
            ground(Eigen::Vector3d norm, Eigen::Vector3d p) ;
            Eigen::Vector3d orient;
            Eigen::Vector3d position;
            ground coordinate(Eigen::Matrix3d R, Eigen::Vector3d p) ; // world frame
            void predict(Eigen::Vector3d dR, double dz) ;
    };
    class lidar {
        public:
            lidar(Eigen::Vector3d offset_) ;
            Eigen::Vector3d measured_point(double dist = 0) ;
            ground body_frame_ground (lidar a, lidar b) ; // body frame
        private:
            Eigen::Vector3d offset; // body frame
            Eigen::Vector3d measured; // body frame
    };
    class leg_states {
        public:
            leg_states(Leg &model) ;
            void calculate(double theta, double theta_d, double beta, double beta_d, Eigen::Vector3d w) ;
            void predict(Eigen::Matrix3d R, ground gnd, double dt) ;
            void assign_contact_point(Eigen::Vector3d p, Eigen::Matrix3d R, RIM rim) ;
            double lookup_omega(RIM rim) ;
            Eigen::Vector3d lookup_point(RIM rim) ;
            Eigen::Vector3d lookup_velocity(RIM rim) ;
            Eigen::Vector3d lookup_predicted_contact_point(RIM rim) ;
            Leg &leg;
        private:
            std::map<RIM, Eigen::Vector3d> rim_center_velocities; // body frame
            std::map<RIM, Eigen::Vector3d> rim_center_points; // body frame
            std::map<RIM, Eigen::Vector3d> predicted_contact_points; // world frame
            std::map<RIM, double> rim_center_omega;
    };
    class states {
        public:
            states(Eigen::Vector3d p_, Eigen::Vector3d v_, 
            leg_states &lf_, leg_states &rf_, leg_states &rh_, leg_states &lh_, 
            lidar &Llf_, lidar &Lrf_, lidar &Lrh_, lidar &Llh_, 
            ground &Glf_, ground &Grf_, ground &Grh_, ground &Glh_) ;
            void predict(Eigen::Vector3d a, Eigen::Matrix3d R, Eigen::Vector3d dba, Eigen::Vector3d dbw, double dt) ;
            double validate_leg(Eigen::Vector3d p, Eigen::Matrix3d R, bool contact, bool slip, leg_states &leg, ground g) ;
            double validate_ground(Eigen::Vector3d p, Eigen::Matrix3d R, ground g, lidar l, lidar rl, lidar fh) ;
            Eigen::Vector3d p; // world frame
            Eigen::Vector3d v; // world frame
            ground &Glf;
            ground &Grf;
            ground &Grh;
            ground &Glh; // world frame
            Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0); 
            Eigen::Vector3d bw = Eigen::Vector3d(0, 0, 0); // body frame
            uint8_t contact_states = 15; // b : slip x 4 bits {0: no slip, 1: slip}; contact x 4 bits {0: swing, 1: contact}
        private:
            leg_states &lf;
            leg_states &rf;
            leg_states &rh;
            leg_states &lh;
            lidar &Llf;
            lidar &Lrf;
            lidar &Lrh;
            lidar &Llh;
            const uint8_t contact_condition = 15; // if contact && contact_condition == 0, which means no leg was estimated to be contact, so weight = 0
    };  
}

#endif