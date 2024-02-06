#include "ParticleFilter.hpp"

namespace estimator {
void generateGaussianVector(Eigen::VectorXd& vector, double mean, double stddev) {
    int N = vector.size();
    std::vector<double> gaussianData(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, stddev);
    std::generate(gaussianData.begin(), gaussianData.end(), [&]() { return distribution(gen); });
    vector = Eigen::Map<Eigen::VectorXd>(gaussianData.data(), N);
}
uint8_t bit_flip_with_prob(uint8_t input, double p) {
    for (int i = 0; i < 8; ++i) {
        double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
        if (randomValue < p) {
            input ^= (1 << i);
        }
    }
}
ParticleFilter::ParticleFilter(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, int number_of_particles, double delta_t) {
    dt = delta_t;
    N = number_of_particles;
    for (int i = 0; i < number_of_particles; i++) {
        Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
        lf_legs.push_back(&lf_leg);
        Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
        rf_legs.push_back(&rf_leg);
        Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
        rh_legs.push_back(&rh_leg);
        Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);
        lh_legs.push_back(&lh_leg);
        lf_leg_states.push_back(*lf_legs[i]);
        rf_leg_states.push_back(*rf_legs[i]);
        rh_leg_states.push_back(*rh_legs[i]);
        lh_leg_states.push_back(*lh_legs[i]);
        lf_leg_states[i].calculate(lf(0), lf(2), lf(1), lf(3), Eigen::Vector3d(0, 0, 0));
        rf_leg_states[i].calculate(rf(0), rf(2), rf(1), rf(3), Eigen::Vector3d(0, 0, 0));
        rh_leg_states[i].calculate(rh(0), rh(2), rh(1), rh(3), Eigen::Vector3d(0, 0, 0));
        lh_leg_states[i].calculate(lh(0), lh(2), lh(1), lh(3), Eigen::Vector3d(0, 0, 0));
        lf_lidar.push_back(lidar(Eigen::Vector3d(0.2 , 0.08, 0)));
        rf_lidar.push_back(lidar(Eigen::Vector3d(0.2 , -0.08, 0)));
        rh_lidar.push_back(lidar(Eigen::Vector3d(-0.2 , -0.08, 0)));
        lh_lidar.push_back(lidar(Eigen::Vector3d(-0.2 , 0.08, 0)));
        lf_ground.push_back(ground(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0)));
        rf_ground.push_back(ground(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0)));
        rh_ground.push_back(ground(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0)));
        lh_ground.push_back(ground(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0)));
        particles.push_back(states(p, v, 
        lf_leg_states[i], rf_leg_states[i], rh_leg_states[i], lh_leg_states[i],
        lf_lidar[i], rf_lidar[i], rh_lidar[i], lh_lidar[i],
        lf_ground[i], rf_ground[i], rh_ground[i], lh_ground[i]));
        weights.push_back(1. / N);
    }
    accel_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-3, 1e-3, 1e-3); // x, y, z
    twist_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-4, 1e-4, 1e-4); // x, y, z
    accel_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-5, 1e-5, 1e-5); // x, y, z
    twist_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-5, 1e-5, 1e-5); // x, y, z
    encoder_noise = Eigen::DiagonalMatrix<double, 4, 4> (1e-2, 1e-2, 1e-1, 1e-1); // theta, beta, theta_d, beta_d
    lidar_noise = Eigen::DiagonalMatrix<double, 4, 4> (1.5e-3, 1.5e-3, 1.5e-3, 1.5e-3); // lf, rf, rh, lh
    attitude_noise = Eigen::DiagonalMatrix<double, 3, 3>(1e-2, 1e-2, 1e-1); // r, p, y
    ground_noise = Eigen::DiagonalMatrix<double, 4, 4>(1e-2, 1e-2, 1e-1, 1e-3); // r, p, y, d
}

void ParticleFilter::lidar_measurement(double lf, double rf, double rh, double lh) {
    for (int i = 0; i < N; i++) {
        Eigen::VectorXd noise_density;
        generateGaussianVector(noise_density);
        Eigen::Vector4d noise = lidar_noise * noise_density;
        lf_lidar[i].measured_point(lf + noise(0));
        rf_lidar[i].measured_point(rf + noise(1));
        rh_lidar[i].measured_point(rh + noise(2));
        lh_lidar[i].measured_point(lh + noise(3));
    }
}

void ParticleFilter::update(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Quaterniond q) {
    Eigen::Matrix3d R = q.toRotationMatrix();
    for (int i = 0; i < N; i++) {
        Eigen::VectorXd accel_noise_density, twist_noise_density,
         accel_bias_noise_density, twist_bias_noise_density, attitude_noise_density;
        Eigen::VectorXd ground_noise_density_lf, ground_noise_density_rf, ground_noise_density_rh, ground_noise_density_lh;
        generateGaussianVector(accel_noise_density);
        generateGaussianVector(twist_noise_density);
        generateGaussianVector(accel_bias_noise_density);
        generateGaussianVector(twist_bias_noise_density);
        generateGaussianVector(attitude_noise_density);
        generateGaussianVector(ground_noise_density_lf);
        generateGaussianVector(ground_noise_density_rf);
        generateGaussianVector(ground_noise_density_rh);
        generateGaussianVector(ground_noise_density_lh);
        Eigen::Vector4d lf_ground_noise = ground_noise * ground_noise_density_lf;
        Eigen::Vector4d rf_ground_noise = ground_noise * ground_noise_density_rf;
        Eigen::Vector4d rh_ground_noise = ground_noise * ground_noise_density_rh;
        Eigen::Vector4d lh_ground_noise = ground_noise * ground_noise_density_lh;

        Eigen::Vector3d delta_attitude = attitude_noise * attitude_noise_density;
        Eigen::Matrix3d m_noise = (Eigen::AngleAxisd(delta_attitude(0), Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(delta_attitude(1), Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(delta_attitude(2), Eigen::Vector3d::UnitZ())).toRotationMatrix();
        
        lf_leg_states[i].predict(m_noise * R, lf_ground[i], dt);
        rf_leg_states[i].predict(m_noise * R, rf_ground[i], dt);
        rh_leg_states[i].predict(m_noise * R, rh_ground[i], dt);
        lh_leg_states[i].predict(m_noise * R, lh_ground[i], dt);

        lf_ground[i].predict(Eigen::Vector3d(lf_ground_noise(0), lf_ground_noise(1), lf_ground_noise(2)), lf_ground_noise(3));
        rf_ground[i].predict(Eigen::Vector3d(rf_ground_noise(0), rf_ground_noise(1), rf_ground_noise(2)), rf_ground_noise(3));
        rh_ground[i].predict(Eigen::Vector3d(rh_ground_noise(0), rh_ground_noise(1), rh_ground_noise(2)), rh_ground_noise(3));
        lh_ground[i].predict(Eigen::Vector3d(lh_ground_noise(0), lh_ground_noise(1), lh_ground_noise(2)), lh_ground_noise(3));

        particles[i].predict(a + accel_noise * accel_noise_density, m_noise * R, accel_bias_noise * accel_bias_noise_density, 
        twist_bias_noise * twist_bias_noise_density, dt);
        particle[i].contact_states = bit_flip_with_prob(particle[i].contact_states, 0.2);
    }
}

void ParticleFilter::calculate_weight(Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, bool update) {
    for (int i = 0; i < N; i++) {
        double weight_i = 1.;
        if (update) {
            validate_ground(Eigen::Vector3d p, Eigen::Matrix3d R, ground g, lidar l, lidar rl, lidar fh) ;
        }
    }
}

void ParticleFilter::resample() {

}

}