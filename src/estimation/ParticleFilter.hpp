#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <Eigen/Dense>
#include "definition.hpp"
#include <random>
namespace estimator {

void generateGaussianVector(Eigen::VectorXd& vector, double mean = 0.0, double stddev = 1.0) ;
uint8_t bit_flip_with_prob(uint8_t input, double p) ;

class ParticleFilter {
    public:
        ParticleFilter(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, int number_of_particles = 50, double delta_t = 0.001) ;
        void lidar_measurement(double lf, double rf, double rh, double lh) ;
        void update(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Quaterniond q) ;
        void calculate_weight(Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, bool update) ;
        void resample() ;
    private:
        std::vector<states> particles;
        std::vector<ground> lf_ground;
        std::vector<ground> rf_ground;
        std::vector<ground> rh_ground;
        std::vector<ground> lh_ground;
        std::vector<leg_states> lf_leg_states;
        std::vector<leg_states> rf_leg_states;
        std::vector<leg_states> rh_leg_states;
        std::vector<leg_states> lh_leg_states;
        std::vector<Leg*> lf_legs;
        std::vector<Leg*> rf_legs;
        std::vector<Leg*> rh_legs;
        std::vector<Leg*> lh_legs;
        std::vector<lidar> lf_lidar;
        std::vector<lidar> rf_lidar;
        std::vector<lidar> rh_lidar;
        std::vector<lidar> lh_lidar;
        std::vector<double> weights;
        Eigen::DiagonalMatrix<double, 3, 3> accel_noise; // x, y, z
        Eigen::DiagonalMatrix<double, 3, 3> twist_noise; // x, y, z
        Eigen::DiagonalMatrix<double, 4, 4> encoder_noise; // theta, beta, theta_d, beta_d
        Eigen::DiagonalMatrix<double, 4, 4> lidar_noise; // lf, rf, rh, lh
        Eigen::DiagonalMatrix<double, 3, 3> attitude_noise; // r, p, y
        Eigen::DiagonalMatrix<double, 4, 4> ground_noise; // r, p, y, d
        Eigen::DiagonalMatrix<double, 3, 3> accel_bias_noise; // x, y, z
        Eigen::DiagonalMatrix<double, 3, 3> twist_bias_noise; // x, y, z
        int N;
        double dt;
};

}

#endif