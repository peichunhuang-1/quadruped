#ifndef PARTICLEFILTER_HPP
#define PARTICLEFILTER_HPP
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ContactMap.hpp"
Eigen::Matrix3d AngleAxisToRotation(Eigen::Vector3d dw) {
    double dtheta = dw.norm();
    Eigen::Vector3d n_dw = dw / dtheta;
    Eigen::Matrix3d skew_dw; skew_dw << 0, -dw(2), dw(1), dw(2), 0, -dw(0), -dw(1), dw(0), 0;
    return Eigen::MatrixXd::Identity(3, 3) + sin(dtheta) * skew_dw + (1 - cos(dtheta)) * (dw * dw.transpose() - Eigen::MatrixXd::Identity(3, 3));
}

struct Particle {
    public:
        Particle() {
            x = State();
            weight = 1.;
            accel_bias = Eigen::Vector3d(0, 0, 0);
            twist_bias = Eigen::Vector3d(0, 0, 0);
        }
        Particle(State xi, ContactInfo info_i) {
            weight = 1.;
            x = xi;
            info = info_i;
            accel_bias = Eigen::Vector3d(0, 0, 0);
            twist_bias = Eigen::Vector3d(0, 0, 0);
        }
        State x;
        ContactInfo info;
        double weight;
        Eigen::Vector3d accel_bias;
        Eigen::Vector3d twist_bias;
};

struct ContactPoint {
    public:
        ContactPoint() {
            point = Eigen::Vector3d(0, 0, 0);
            state = 0;
            rim_beta = 0.0;
        }
        Eigen::Vector3d point;
        double state;
        double rim_beta;
};

class ParticleFilter {
    public:
        ParticleFilter(const size_t n, 
        Eigen::Vector3d a_sigma, Eigen::Vector3d w_sigma, Eigen::Vector4d input_sigma, Eigen::Vector3d v_sigma, 
        Eigen::Vector4d measure_sigma,
        double R = 0.1, double r = 0.01, double flip = 3., double ground_sigma = 1e-1) {
            N = n;
            accel_sigma = a_sigma;
            omega_sigma = w_sigma;
            motor_input_sigma = input_sigma;
            measurement_sigma = measure_sigma;
            velocity_sigma = v_sigma;
            ground_noise = ground_sigma;
            flip_prob = flip;
            Nth = n / 2.;
            for (int i = 0; i < 4; i++) particles[i].resize(N);
            this->legs.push_back(Leg(Eigen::Vector3d(0.2, 0.15, 0.), R, r));
            this->legs.push_back(Leg(Eigen::Vector3d(0.2, -0.15, 0.), R, r));
            this->legs.push_back(Leg(Eigen::Vector3d(-0.2, -0.15, 0.), R, r));
            this->legs.push_back(Leg(Eigen::Vector3d(-0.2, 0.15, 0.), R, r));
        }

        void Initialize(State x_init, Eigen::Matrix4d encoders_init) {
            for (int j = 0; j < 4; j++) {
                Eigen::Vector4d encoder = encoders_init.row(j);
                this->legs[j].Calculate(encoder(0), encoder(2), 0, encoder(1), encoder(3), 0);
            }
            for (size_t i = 0; i < N; i++) {
                for (int j = 0; j < 4; j++) {
                    Eigen::Vector4d encoder = encoders_init.row(j);
                    ContactInfo contact_measure = map.ContactPoint(encoder(0), encoder(2), encoder(1), encoder(3), Eigen::Vector3d(0, 0, 0), x_init, this->legs[j]);
                    particles[j][i].x = x_init;
                    particles[j][i].info = contact_measure;
                }
            }
            x = x_init;
            for (int j = 0; j < 4; j++) {
                Eigen::Vector4d encoder = encoders_init.row(j);
                ContactInfo contact_measure = map.ContactPoint(encoder(0), encoder(2), encoder(1), encoder(3), Eigen::Vector3d(0, 0, 0), x_init, this->legs[j]);
                contact_points[j].point = contact_measure.point_contact;
            }
        }

        void Prediction(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Matrix4d encoders, double dt, Eigen::Vector<bool, 4> contact) { // encoders: theta, beta, theta_d, beta_d , this encoder data is at k-1 time tick
            for (size_t i = 0; i < N; i++) {
                Eigen::Vector3d a_with_noise = a + accel_sigma * ((rand() - RAND_MAX / 2.) / RAND_MAX) - Eigen::Vector3d(1e-2, 1e-2, 1e-2);
                Eigen::Vector3d w_with_noise = w + omega_sigma * ((rand() - RAND_MAX / 2.) / RAND_MAX) - Eigen::Vector3d(0.001, 0.001, 0.001);
                Eigen::Vector3d v_noise = velocity_sigma * ((rand() - RAND_MAX / 2.) / RAND_MAX);
                Eigen::Matrix3d dR = AngleAxisToRotation(w_with_noise * dt);
                for (int j = 0; j < 4; j++) {
                    Eigen::Quaterniond q(particles[j][i].x.quaternion);
                    q.normalize();
                    Eigen::Matrix3d last_R = q.toRotationMatrix();
                    Eigen::Vector3d position = particles[j][i].x.position + last_R * (particles[j][i].x.velocity * dt + a_with_noise * dt * dt * 0.5);
                    Eigen::Vector3d velocity = dR * (particles[j][i].x.velocity + a_with_noise * dt + v_noise);
                    Eigen::Matrix3d R = dR * last_R;
                    Eigen::Vector4d encoder = encoders.row(j);
                    Eigen::Vector4d encoders_with_noise = encoder + ((rand() - RAND_MAX / 2.) / RAND_MAX) * motor_input_sigma;
                    particles[j][i].info.state = rand() > (RAND_MAX / flip_prob)? particles[j][i].info.state : ~particles[j][i].info.state; // 1 swing, 0 contact
                    particles[j][i].info.rim_beta += ((rand() - RAND_MAX / 2.) / RAND_MAX) * ground_noise;
                    this->legs[j].Calculate(encoders_with_noise(0), encoders_with_noise(2), 0, encoders_with_noise(1), encoders_with_noise(3), 0);
                    // if (!particles[j][i].info.state) {
                    if (!contact(j)) {
                        particles[j][i].info.point_contact = particles[j][i].info.point_contact + Eigen::Vector3d(1e-4, 1e-4, 1e-4) * ((rand() - RAND_MAX / 2.) / RAND_MAX);
                        // particles[j][i].info.rim_beta = contact_points[j].rim_beta;
                        particles[j][i].info = map.RollingContact(encoders_with_noise(2), encoders_with_noise(3), w_with_noise, particles[j][i].x, particles[j][i].info, this->legs[j], dt);
                    }
                    particles[j][i].x.position = position;
                    particles[j][i].x.velocity = velocity;
                    particles[j][i].x.quaternion = Eigen::Quaterniond(R).coeffs();
                }
            }
        }
        void Observation(Eigen::Vector3d w, Eigen::Matrix4d encoders, Eigen::Vector<bool, 4> contact) {
            for (int j = 0; j < 4; j++) {
                Eigen::Vector4d encoder = encoders.row(j);
                this->legs[j].Calculate(encoder(0), encoder(2), 0, encoder(1), encoder(3), 0);
            }
            summation = Eigen::Vector4d(0, 0, 0, 0);
            summation_square = Eigen::Vector4d(0, 0, 0, 0);
            for (size_t i = 0; i < N; i++) {
                for (int j = 0; j < 4; j++) {
                    Eigen::Vector4d encoder = encoders.row(j);
                    ContactInfo contact_measure = map.ContactPoint(encoder(0), encoder(2), encoder(1), encoder(3), w, particles[j][i].x, this->legs[j]);
                    contact_measure.state = particles[j][i].info.state;
                    Eigen::Vector4d sigma = measurement_sigma;
                    // if (particles[j][i].info.state) {
                    if (contact(j)) {
                        particles[j][i].info = contact_measure;
                        // particles[j][i].info.point_velocity = Eigen::Vector3d(0, 0, 0);
                        sigma *= exp(100);
                    }
                    Eigen::Vector3d contact_err = particles[j][i].info.point_contact - contact_measure.point_contact;
                    Eigen::Vector4d error(contact_err(0), contact_err(1), contact_err(2), particles[j][i].info.point_velocity(2));
                    // Eigen::Vector4d error(particles[j][i].info.point_velocity(0), particles[j][i].info.point_velocity(1), particles[j][i].info.point_velocity(2), 0);
                    Eigen::Vector4d sigma_square = sigma.cwiseProduct(sigma);
                    Eigen::DiagonalMatrix<double, 4, 4> residual_matrix( sigma_square(0), sigma_square(1), sigma_square(2), sigma_square(3));
                    double exponent = -0.5 * error.transpose() * residual_matrix.inverse() * error;
                    double normalization = std::pow(2.0 * M_PI, -2) * std::pow(residual_matrix.toDenseMatrix().determinant(), -0.5);
                    double weight_ = normalization * std::exp(exponent);
                    particles[j][i].weight *= weight_;
                    // if (contact(j)) {
                    //     particles[j][i].weight = 1. / (double) N;
                    // }
                    summation(j) += particles[j][i].weight;
                }
            }
            for (size_t j = 0; j < 4; j++) {
                if (summation(j) <= 1e-8) {
                    std::cout << "result not accurate\n";
                    summation(j) = 1e-8;
                    for (size_t i = 0; i < N; i++) {
                        particles[j][i].weight = 1e-8 / (double) N;
                    }
                }
            }
            double sum = summation(0) + summation(1) + summation(2) + summation(3);
            x = State();
            x.quaternion = Eigen::Vector4d(0, 0, 0, 0);
            for (int k = 0; k < 4; k ++ ) contact_points[k] = ContactPoint();
            for (size_t i = 0; i < N; i++) {
                for (int j = 0; j < 4; j++) {
                    x.position += (particles[j][i].weight * particles[j][i].x.position / sum);
                    x.velocity += (particles[j][i].weight * particles[j][i].x.velocity / sum);
                    x.quaternion += (particles[j][i].weight * particles[j][i].x.quaternion / sum);
                    contact_points[j].point += (particles[j][i].weight * particles[j][i].info.point_contact / summation(j));
                    contact_points[j].state += (particles[j][i].weight * ((double) particles[j][i].info.state) / summation(j));
                    contact_points[j].rim_beta += (particles[j][i].weight * particles[j][i].info.rim_beta / summation(j));
                    particles[j][i].weight /= summation(j);
                    summation_square(j) += (particles[j][i].weight * particles[j][i].weight);
                }
            }
            for (int j = 0; j < 4; j++) {
                if (1. / summation_square(j) < this->Nth) {
                    std::cout << "resample" << "\n";
                    Resample(j);
                }
                // if (summation(j) <= 1e-8) {
                //     std::cout << "resample" << "\n";
                //     Resample(j);
                // }
            }
        }
        void Resample(int j) {
            int ind = 0;
            Eigen::VectorXd important_index = Eigen::VectorXd::Zero(N);
            double step = 1. / (double) N;
            for (int i = 0; i < N; i++) {
                while (((double) i) * step > particles[j][ind].weight){
                    if (ind >= N - 1) break;
                    ind += 1;
                }
                important_index(i) = ind;
            }
            for (int i = 0; i < N; i++) {
                particles[j][i] = particles[j][important_index(i)];
                particles[j][i].weight = step;
            }
        }
        State state() {return x;};
        ContactPoint contact_point(int k) {return contact_points[k];}

    private:
        size_t N;
        double Nth;
        Eigen::Vector3d accel_sigma;
        Eigen::Vector3d omega_sigma;
        Eigen::Vector4d motor_input_sigma;
        Eigen::Vector4d measurement_sigma;
        double ground_noise;
        double flip_prob;
        std::vector<Leg> legs;
        std::vector<Particle> particles[4];
        ContactMap map;
        State x;
        ContactPoint contact_points[4];
        Eigen::Vector4d summation;
        Eigen::Vector4d summation_square;
        Eigen::Vector3d velocity_sigma;
};

#endif