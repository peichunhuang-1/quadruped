#ifndef DEFINITION_HPP
#define DEFINITION_HPP
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "ContactMap.hpp"
namespace estimatior {
    class ground {
        public:
            ground() ;
            ground(Eigen::Vector3d norm, Eigen::Vector3d p) ;
            Eigen::Vector3d orient;
            Eigen::Vector3d position;
            ground coordinate(Eigen::Quaterniond q, Eigen::Vector3d p) ;
    };
    class lidar {
        public:
            lidar(Eigen::Vector3d offset_) ;
            Eigen::Vector3d measured_point(double dist = 0) ;
            ground body_frame_ground (lidar a, lidar b) ;
        private:
            Eigen::Vector3d offset;
            Eigen::Vector3d measured;
    };
    class leg_states {
        public:
            leg_states(Leg &model) ;
            void calculate(double theta, double theta_d, double beta, double beta_d) ;
            void predict(ground gnd, double dt) ;
            double lookup_omega(RIM rim) ;
            Eigen::Vector3d lookup_point(RIM rim) ;
            Eigen::Vector3d lookup_velocity(RIM rim) ;
            Eigen::Vector3d lookup_predicted_contact_point(RIM rim) ;
        private:
            Leg &leg;
            std::map<RIM, Eigen::Vector3d> rim_center_velocities;
            std::map<RIM, Eigen::Vector3d> rim_center_points;
            std::map<RIM, Eigen::Vector3d> predicted_contact_points;
            std::map<RIM, double> rim_center_omega;
    };
    class states {
        public:
            states(Eigen::Vector3d p_, Eigen::Vector3d v_, 
            leg_states &lf_, leg_states &rf_, leg_states &rh_, leg_states &lh_, 
            lidar &Llf_, lidar &Lrf_, lidar &Lrh_, lidar &Llh_) ;
            void predict(Eigen::Vector3d a, Eigen::Quaterniond q, double dt) ;
            void validation() ;
            Eigen::Vector3d p;
            Eigen::Vector3d v;
            ground Glf;
            ground Grf;
            ground Grh;
            ground Glh;
            Eigen::Vector3d ba = 0;
            Eigen::Vector3d bw = 0;
        private:
            leg_states &lf;
            leg_states &rf;
            leg_states &rh;
            leg_states &lh;
            lidar &Llf;
            lidar &Lrf;
            lidar &Lrh;
            lidar &Llh;
    };  
}

#endif