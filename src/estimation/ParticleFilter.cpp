#include "ParticleFilter.hpp"

namespace estimator {

template<int N>
void generateGaussianVector(Eigen::Vector<double, N>& vector, double mean, double stddev) {
    std::vector<double> gaussianData(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, stddev);
    std::generate(gaussianData.begin(), gaussianData.end(), [&]() { return distribution(gen); });
    vector = Eigen::Map<Eigen::Vector<double, N>>(gaussianData.data(), N);
}
uint8_t bit_flip_with_prob(uint8_t input, double p) {
    for (int i = 0; i < 8; ++i) {
        double randomValue = static_cast<double>(std::rand()) / RAND_MAX;
        if (randomValue < p) {
            input ^= (1 << i);
        }
    }
    return input;
}

ParticleState::ParticleState() {
    this->contact_expr = {0, 0, 0, 0, 0, 0, 0, 0};
    this->p = Eigen::Vector3d(0, 0, 0);
    this->v = Eigen::Vector3d(0, 0, 0);
    this->ba = Eigen::Vector3d(0, 0, 0);
    this->bw = Eigen::Vector3d(0, 0, 0);
    this->C_lf = Eigen::Vector3d(0, 0, 0);
    this->C_rf = Eigen::Vector3d(0, 0, 0);
    this->C_rh = Eigen::Vector3d(0, 0, 0);
    this->C_lh = Eigen::Vector3d(0, 0, 0);
    this->G_lf.orient = Eigen::Vector3d(0, 0, 0);
    this->G_rf.orient = Eigen::Vector3d(0, 0, 0);
    this->G_rh.orient = Eigen::Vector3d(0, 0, 0);
    this->G_lh.orient = Eigen::Vector3d(0, 0, 0);
    this->G_lf.position = Eigen::Vector3d(0, 0, 0);
    this->G_rf.position = Eigen::Vector3d(0, 0, 0);
    this->G_rh.position = Eigen::Vector3d(0, 0, 0);
    this->G_lh.position = Eigen::Vector3d(0, 0, 0);
}
ParticleState ParticleState::operator += (const std::pair<double, states > &s) {
    
    this->p += s.first * s.second.p;
    this->v += s.first * s.second.v;
    this->ba += s.first * s.second.ba;
    this->bw += s.first * s.second.bw;
    double lf_radius = s.second.lf->rim_contact == G_POINT? s.second.lf->leg->radius() : s.second.lf->leg->Radius() + s.second.lf->leg->radius();
    double rf_radius = s.second.rf->rim_contact == G_POINT? s.second.rf->leg->radius() : s.second.rf->leg->Radius() + s.second.rf->leg->radius();
    double rh_radius = s.second.rh->rim_contact == G_POINT? s.second.rh->leg->radius() : s.second.rh->leg->Radius() + s.second.rh->leg->radius();
    double lh_radius = s.second.lh->rim_contact == G_POINT? s.second.lh->leg->radius() : s.second.lh->leg->Radius() + s.second.lh->leg->radius();
    C_lf += s.first * (s.second.lf->lookup_predicted_contact_point(s.second.lf->rim_contact) - lf_radius * s.second.Glf->orient);
    C_rf += s.first * (s.second.rf->lookup_predicted_contact_point(s.second.rf->rim_contact) - rf_radius * s.second.Grf->orient);
    C_rh += s.first * (s.second.rh->lookup_predicted_contact_point(s.second.rh->rim_contact) - rh_radius * s.second.Grh->orient);
    C_lh += s.first * (s.second.lh->lookup_predicted_contact_point(s.second.lh->rim_contact) - lh_radius * s.second.Glh->orient);
    
    for (int i = 0; i < 8; i++) {
        this->contact_expr[i] += s.first * (double) ((s.second.contact_states >> i) & 0x01);
    }
    this->G_lf.orient += s.first * s.second.Glf->orient;
    this->G_rf.orient += s.first * s.second.Grf->orient;
    this->G_rh.orient += s.first * s.second.Grh->orient;
    this->G_lh.orient += s.first * s.second.Glh->orient;
    this->G_lf.position += s.first * s.second.Glf->position;
    this->G_rf.position += s.first * s.second.Grf->position;
    this->G_rh.position += s.first * s.second.Grh->position;
    this->G_lh.position += s.first * s.second.Glh->position;
    return *this;
}

std::ostream& operator<<(std::ostream& os, const ParticleState &ps) {
    os << "position: \t" << ps.p.transpose() << "\n";
    os << "velocity: \t" << ps.v.transpose() << "\n";
    os << "bias acceleration: \t" << ps.ba.transpose() << "\n";
    os << "bias twist: \t" << ps.bw.transpose() << "\n";
    os << "contact LF: \t" << ps.C_lf.transpose() << "\n";
    os << "contact RF: \t" << ps.C_rf.transpose() << "\n";
    os << "contact RH: \t" << ps.C_rh.transpose() << "\n";
    os << "contact LH: \t" << ps.C_lh.transpose() << "\n";
    os << "ground LF: \t" << ps.G_lf.position.transpose() << "\n";
    os << "ground RF: \t" << ps.G_rf.position.transpose() << "\n";
    os << "ground RH: \t" << ps.G_rh.position.transpose() << "\n";
    os << "ground LH: \t" << ps.G_lh.position.transpose() << "\n";
    os << "ground orient LF: \t" << ps.G_lf.orient.transpose() << "\n";
    os << "ground orient RF: \t" << ps.G_rf.orient.transpose() << "\n";
    os << "ground orient RH: \t" << ps.G_rh.orient.transpose() << "\n";
    os << "ground orient LH: \t" << ps.G_lh.orient.transpose() << "\n";
    os << "contact expression: \t";
    for (int i = 7; i >= 0; i--) 
        os << ps.contact_expr[i] << "\t";
    os << "\n";
    return os;
}

ParticleFilter::ParticleFilter(Eigen::Vector3d p, Eigen::Vector3d v, Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, int number_of_particles, double delta_t) {
    dt = delta_t;
    N = number_of_particles;
    lf_legs.resize(number_of_particles);
    rf_legs.resize(number_of_particles);
    rh_legs.resize(number_of_particles);
    lh_legs.resize(number_of_particles);
    lf_leg_states.resize(number_of_particles);
    rf_leg_states.resize(number_of_particles);
    rh_leg_states.resize(number_of_particles);
    lh_leg_states.resize(number_of_particles);
    lf_lidar.resize(number_of_particles);
    rf_lidar.resize(number_of_particles);
    rh_lidar.resize(number_of_particles);
    lh_lidar.resize(number_of_particles);
    lf_ground.resize(number_of_particles);
    rf_ground.resize(number_of_particles);
    rh_ground.resize(number_of_particles);
    lh_ground.resize(number_of_particles);

    for (int i = 0; i < number_of_particles; i++) {
        lf_legs[i] = std::make_shared<Leg>(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
        rf_legs[i] = std::make_shared<Leg>(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
        rh_legs[i] = std::make_shared<Leg>(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
        lh_legs[i] = std::make_shared<Leg>(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);

        lf_leg_states[i] = std::make_shared<leg_states>(lf_legs[i]);
        rf_leg_states[i] = std::make_shared<leg_states>(rf_legs[i]);
        rh_leg_states[i] = std::make_shared<leg_states>(rh_legs[i]);
        lh_leg_states[i] = std::make_shared<leg_states>(lh_legs[i]);

        lf_leg_states[i]->calculate(lf(0), lf(2), lf(1), lf(3), Eigen::Vector3d(0, 0, 0));
        rf_leg_states[i]->calculate(rf(0), rf(2), rf(1), rf(3), Eigen::Vector3d(0, 0, 0));
        rh_leg_states[i]->calculate(rh(0), rh(2), rh(1), rh(3), Eigen::Vector3d(0, 0, 0));
        lh_leg_states[i]->calculate(lh(0), lh(2), lh(1), lh(3), Eigen::Vector3d(0, 0, 0));

        lf_leg_states[i]->init(p, Eigen::Matrix3d::Identity());
        rf_leg_states[i]->init(p, Eigen::Matrix3d::Identity());
        rh_leg_states[i]->init(p, Eigen::Matrix3d::Identity());
        lh_leg_states[i]->init(p, Eigen::Matrix3d::Identity());

        lf_lidar[i] = std::make_shared<lidar>(Eigen::Vector3d(0.2 , 0.08, 0));
        rf_lidar[i] = std::make_shared<lidar>(Eigen::Vector3d(0.2 , -0.08, 0));
        rh_lidar[i] = std::make_shared<lidar>(Eigen::Vector3d(-0.2 , -0.08, 0));
        lh_lidar[i] = std::make_shared<lidar>(Eigen::Vector3d(-0.2 , 0.08, 0));
        lf_ground[i] = std::make_shared<ground>(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
        rf_ground[i] = std::make_shared<ground>(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
        rh_ground[i] = std::make_shared<ground>(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
        lh_ground[i] = std::make_shared<ground>(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 0, 0));
        particles.push_back(states(p, v, 
        lf_leg_states[i], rf_leg_states[i], rh_leg_states[i], lh_leg_states[i],
        lf_lidar[i], rf_lidar[i], rh_lidar[i], lh_lidar[i],
        lf_ground[i], rf_ground[i], rh_ground[i], lh_ground[i]));
        weights.push_back(1. / (double) N);
    }

    accel_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-2, 1e-2, 1e-2); // x, y, z
    twist_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-3, 1e-3, 1e-3); // x, y, z
    accel_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-5, 1e-5, 1e-5); // x, y, z
    twist_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (1e-5, 1e-5, 1e-5); // x, y, z
    encoder_noise = Eigen::DiagonalMatrix<double, 4, 4> (3e-2, 3e-2, 3e-1, 3e-1); // theta, beta, theta_d, beta_d
    lidar_noise = Eigen::DiagonalMatrix<double, 4, 4> (5e-3, 5e-3, 5e-3, 5e-3); // lf, rf, rh, lh
    attitude_noise = Eigen::DiagonalMatrix<double, 3, 3>(1e-2, 1e-2, 1e-1); // r, p, y
    ground_noise = Eigen::DiagonalMatrix<double, 4, 4>(1e-4, 1e-4, 1e-2, 1e-4); // r, p, y, d
    // accel_noise = Eigen::DiagonalMatrix<double, 3, 3> (0, 0, 0); // x, y, z
    // twist_noise = Eigen::DiagonalMatrix<double, 3, 3> (0, 0, 0); // x, y, z
    // accel_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (0, 0, 0); // x, y, z
    // twist_bias_noise = Eigen::DiagonalMatrix<double, 3, 3> (0, 0, 0); // x, y, z
    // encoder_noise = Eigen::DiagonalMatrix<double, 4, 4> (0, 0, 0, 0); // theta, beta, theta_d, beta_d
    // lidar_noise = Eigen::DiagonalMatrix<double, 4, 4> (0, 0, 0, 0); // lf, rf, rh, lh
    // attitude_noise = Eigen::DiagonalMatrix<double, 3, 3>(0, 0, 0); // r, p, y
    // ground_noise = Eigen::DiagonalMatrix<double, 4, 4>(0, 0, 0, 0); // r, p, y, d
}

void ParticleFilter::lidar_measurement(double lf, double rf, double rh, double lh) {
    for (int i = 0; i < N; i++) {
        Eigen::Vector4d noise_density;
        generateGaussianVector<4>(noise_density);
        Eigen::Vector4d noise = lidar_noise * noise_density;
        lf_lidar[i]->measured_point(lf + noise(0));
        rf_lidar[i]->measured_point(rf + noise(1));
        rh_lidar[i]->measured_point(rh + noise(2));
        lh_lidar[i]->measured_point(lh + noise(3));
    }
}

void ParticleFilter::update(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Quaterniond q) {
    Eigen::Matrix3d R = q.toRotationMatrix();
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d accel_noise_density, twist_noise_density,
         accel_bias_noise_density, twist_bias_noise_density, attitude_noise_density;
        Eigen::Vector4d ground_noise_density_lf, ground_noise_density_rf, ground_noise_density_rh, ground_noise_density_lh;
        generateGaussianVector<3>(accel_noise_density);
        generateGaussianVector<3>(twist_noise_density);
        generateGaussianVector<3>(accel_bias_noise_density);
        generateGaussianVector<3>(twist_bias_noise_density);
        generateGaussianVector<3>(attitude_noise_density);
        generateGaussianVector<4>(ground_noise_density_lf);
        generateGaussianVector<4>(ground_noise_density_rf);
        generateGaussianVector<4>(ground_noise_density_rh);
        generateGaussianVector<4>(ground_noise_density_lh);

        Eigen::Vector4d lf_ground_noise = ground_noise * ground_noise_density_lf;
        Eigen::Vector4d rf_ground_noise = ground_noise * ground_noise_density_rf;
        Eigen::Vector4d rh_ground_noise = ground_noise * ground_noise_density_rh;
        Eigen::Vector4d lh_ground_noise = ground_noise * ground_noise_density_lh;

        Eigen::Vector3d delta_attitude = attitude_noise * attitude_noise_density;
        Eigen::Matrix3d m_noise = (Eigen::AngleAxisd(delta_attitude(0), Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(delta_attitude(1), Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(delta_attitude(2), Eigen::Vector3d::UnitZ())).toRotationMatrix();

        lf_ground[i]->predict(Eigen::Vector3d(lf_ground_noise(0), lf_ground_noise(1), lf_ground_noise(2)), lf_ground_noise(3));
        rf_ground[i]->predict(Eigen::Vector3d(rf_ground_noise(0), rf_ground_noise(1), rf_ground_noise(2)), rf_ground_noise(3));
        rh_ground[i]->predict(Eigen::Vector3d(rh_ground_noise(0), rh_ground_noise(1), rh_ground_noise(2)), rh_ground_noise(3));
        lh_ground[i]->predict(Eigen::Vector3d(lh_ground_noise(0), lh_ground_noise(1), lh_ground_noise(2)), lh_ground_noise(3));
        
        particles[i].contact_states = bit_flip_with_prob(particles[i].contact_states, 0.2);
        particles[i].predict(a + accel_noise * accel_noise_density, w + twist_noise * twist_noise_density, m_noise * R, accel_bias_noise * accel_bias_noise_density, 
        twist_bias_noise * twist_bias_noise_density, dt);
    }
}

void ParticleFilter::calculate_weight(Eigen::Vector4d lf, Eigen::Vector4d rf, Eigen::Vector4d rh, Eigen::Vector4d lh, bool update) {
    double sum = 0;
    for (int i = 0; i < N; i++) {
        double weight_i = 1.;
        if (update) {
            double g_weight_lf = particles[i].validate_ground(particles[i].p, particles[i].Rotation, *(particles[i].Glf), *(lf_lidar[i]), *(rf_lidar[i]), *(lh_lidar[i])) ;
            double g_weight_rf = particles[i].validate_ground(particles[i].p, particles[i].Rotation, *(particles[i].Grf), *(rf_lidar[i]), *(lf_lidar[i]), *(rh_lidar[i])) ;
            double g_weight_rh = particles[i].validate_ground(particles[i].p, particles[i].Rotation, *(particles[i].Grh), *(rh_lidar[i]), *(lh_lidar[i]), *(rf_lidar[i])) ;
            double g_weight_lh = particles[i].validate_ground(particles[i].p, particles[i].Rotation, *(particles[i].Glh), *(lh_lidar[i]), *(rh_lidar[i]), *(lf_lidar[i])) ;
            weight_i *= (g_weight_lf + g_weight_rf + g_weight_rh + g_weight_lh);
        }
        lf_leg_states[i]->calculate(lf(0), lf(2), lf(1), lf(3), particles[i].omega);
        rf_leg_states[i]->calculate(rf(0), rf(2), rf(1), rf(3), particles[i].omega);
        rh_leg_states[i]->calculate(rh(0), rh(2), rh(1), rh(3), particles[i].omega);
        lh_leg_states[i]->calculate(lh(0), lh(2), lh(1), lh(3), particles[i].omega);
        double leg_weight_lf = particles[i].validate_leg(particles[i].p, particles[i].Rotation, particles[i].contact_states & 0x08, particles[i].contact_states & 0x80, lf_leg_states[i], *(particles[i].Glf)) ;
        double leg_weight_rf = particles[i].validate_leg(particles[i].p, particles[i].Rotation, particles[i].contact_states & 0x04, particles[i].contact_states & 0x40, rf_leg_states[i], *(particles[i].Grf)) ;
        double leg_weight_rh = particles[i].validate_leg(particles[i].p, particles[i].Rotation, particles[i].contact_states & 0x02, particles[i].contact_states & 0x20, rh_leg_states[i], *(particles[i].Grh)) ;
        double leg_weight_lh = particles[i].validate_leg(particles[i].p, particles[i].Rotation, particles[i].contact_states & 0x01, particles[i].contact_states & 0x10, lh_leg_states[i], *(particles[i].Glf)) ;
        weight_i *= (leg_weight_lf + leg_weight_rf + leg_weight_rh + leg_weight_lh);
        if (!particles[i].condition()) weights[i] = 0;
        else weights[i] = weight_i;
        sum += weights[i];
    }

    for (int i = 0; i < N; i++) {
        if (sum == 0) weights[i] = 1. / (double) N;
        else weights[i] = weights[i] / sum;
    }
}

void ParticleFilter::resample() {
    double den = 0.;
    for (int i = 0; i < N; i++) {
        den += (weights[i] * weights[i]);
    }
    double Neff = 1. / den;
    std::cout << "Neff: " << Neff << "\n\n";
    if (Neff >= this->Nth) return;
    std::cout << "resample\n";
    Eigen::VectorXd resample_id = Eigen::VectorXd::Zero(N);
    resample_id += Eigen::VectorXd::Constant(N, 1. / (double) N);
    this->cumsum(resample_id);
    resample_id += (Eigen::VectorXd::Random(N) / (double) N);
    this->cumsum(this->weights);
    Eigen::VectorXd important_index = Eigen::VectorXd::Zero(N);
    int ind = 0;
    for (int i = 0; i < N; i++) {
        while (resample_id(i) > this->weights[ind]){
            if (ind >= N - 1) break;
            ind += 1;
        }
        important_index(i) = ind;
    }
    for (int i = 0; i < N; i++){
        this->particles[important_index(i)].copyto(this->particles[i]);
        this->weights[i] = 1. / (double) N;
    }
}

void ParticleFilter::cumsum(std::vector<double> &arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++)
        arr[i] += arr[i-1];
}

void ParticleFilter::cumsum(Eigen::VectorXd &arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++)
        arr(i) += arr(i-1);
}

ParticleState ParticleFilter::value() {
    ParticleState ps;
    for (int i = 0; i < N; i++) 
        ps += std::pair<double, states >(weights[i], particles[i]);
    return ps;
}

}