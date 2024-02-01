#include "Estimator.hpp"
double gaussian_erf(double x) {
    return (std::erf(x) + 1) * 0.5;
}

VelocityEstimateByLeg::VelocityEstimateByLeg(Eigen::Vector3d offset, Eigen::Vector3d lidar_offset, double R, double r, double dt, int n) : leg(offset, R, r), durations(dt), lidar(lidar_offset) {
    update = true;
    for (int i = 0; i < n; i++) {
        STATE s;
        states.push_back(s);
    }
}

void VelocityEstimateByLeg::calculate(IMU_DATA imu, ENCODER_DATA m, LIDAR_DATA lidar, bool update_) {
    double alpha = lidar.pitch;
    if (! update_) { 
        contact_beta += (imu.w(1)+ m.beta_d) * durations ; 
        alpha = contact_beta - m.beta ;
    }
    else {
        contact_beta = alpha + m.beta;
    }
    ContactMap cm;
    this->leg.Calculate(m.theta, m.theta_d, 0, m.beta, m.beta_d, 0);
    this->leg.PointContact(cm.lookup(m.theta, m.beta + alpha), alpha);
    this->leg.PointVelocity(Eigen::Vector3d(0, 0, 0), imu.w, cm.lookup(m.theta, m.beta + alpha), alpha);
    Eigen::Vector3d point_velocity = - this->leg.contact_velocity;
    STATE state = {
        point_velocity,
        this->leg.contact_point,
        2.5e-3 * Eigen::Matrix3d::Identity(),
        1e-4 * Eigen::Matrix3d::Identity(),
    };
    update = update_ ;
    current_imu = imu ;
    current_m = m ;
    states.pop_front();
    states.push_back(state);
    if (update_) current_lidar = lidar ;
}

BodyEstimation::BodyEstimation(VelocityEstimateByLeg *lf, VelocityEstimateByLeg *rf, VelocityEstimateByLeg *rh, VelocityEstimateByLeg *lh, int n) : N(n) {
    legs.push_back(lf);
    legs.push_back(rf);
    legs.push_back(rh);
    legs.push_back(lh);
    for (int i = 0; i < N; i++) {
        position.push_back(Eigen::Vector3d(0, 0, 0.11));
        velocity.push_back(Eigen::Vector3d(0, 0, 0));
        weights.push_back(Eigen::Vector4d(.5, .5, .5, .5));
        rotation.push_back(Eigen::Matrix3d::Identity());
    }
    sigma = 1e-2;
}

void BodyEstimation::calculate_ground(int index) {
    if (! legs[index]->update) return;
    Eigen::Matrix3d gnd_rotation, gnd_roll, gnd_pitch;
    gnd_pitch << cos(-legs[index]->current_lidar.pitch), 0, sin(-legs[index]->current_lidar.pitch), 0, 1, 0, -sin(-legs[index]->current_lidar.pitch), 0, cos(-legs[index]->current_lidar.pitch);
    gnd_roll << 1, 0, 0, 0, cos(-legs[index]->current_lidar.roll), -sin(-legs[index]->current_lidar.roll), 0, sin(-legs[index]->current_lidar.roll), cos(-legs[index]->current_lidar.roll);
    gnd_rotation = gnd_pitch * gnd_roll ;
    legs[index]->current_gnd = {
        position.back() + rotation.back() * (legs[index]->lidar + Eigen::Vector3d(0, 0, -legs[index]->current_lidar.distance)), 
        rotation.back() * gnd_rotation
    };
}

void BodyEstimation::calculate_weight() {
    Eigen::Vector4d push_weight;
    weights.pop_front();
    for (int i = 0; i < 4; i ++) {
        calculate_ground(i);
        Eigen::Vector3d point_of_contact = position.back() + rotation.back() * (legs[i]->leg.contact_point);
        Eigen::Vector3d N = legs[i]->current_gnd.rotation.transpose().row(2);
        double dst = (-N.dot(point_of_contact) + N.dot(legs[i]->current_gnd.point)) / (N.norm());
        push_weight(i) = gaussian_erf(dst / sqrt(2 * sigma * sigma));
    }
    weights.push_back(push_weight);
}

void BodyEstimation::prediction(Eigen::Matrix3d rot) {
    position.pop_front();
    velocity.pop_front();
    rotation.pop_front();
    position.push_back(position.back() + velocity.back() * legs[0]->durations + 0.5 * legs[0]->durations * legs[0]->durations * rotation.back() * legs[0]->current_imu.a);
    velocity.push_back(velocity.back() + legs[0]->durations * rotation.back() * legs[0]->current_imu.a);
    rotation.push_back(rot);
}

void BodyEstimation::assign(std::vector<double> x) {
    for (int j = 0; j < N; j++) {
        velocity[j] = Eigen::Vector3d(x[3 * j + 0], x[3 * j + 1], x[3 * j + 2]);
    }
}

double opt_func(const std::vector<double> &x, std::vector<double> &grad, void *f_data) { 
    // vx_1, vy_1, vz_1 ... vx_n, vy_n, vz_n
    BodyEstimation *fop = (BodyEstimation*) (f_data);
    Eigen::Matrix3d predicted_cov = Eigen::Matrix3d::Identity() * 1e-4;
    double cost = 0;
    Eigen::Vector3d v;
    for (int j = 0; j < fop->N; j++) {
        v = Eigen::Vector3d(x[3 * j + 0], x[3 * j + 1], x[3 * j + 2]);
        Eigen::Vector4d weights = fop->weights[j];
        for (int i = 0; i < 4; i++) {
            Eigen::Vector3d diff = v - fop->rotation[j] * fop->legs[i]->states[j].predicted_velocity;
            cost += weights(i) * std::sqrt(diff.transpose() * fop->legs[i]->states[j].velocity_covariance.inverse() * diff);
        }
        Eigen::Vector3d diff = v - fop->velocity[j];
        cost += std::sqrt(diff.transpose() * predicted_cov.inverse() * diff);
    }
    return cost;
}

nlopt::opt Optimizer(BodyEstimation* fop) {
    // LN_BOBYQA
    // LN_COBYLA
    // LN_PRAXIS
    // LN_SBPLX
    // GN_CRS2_LM
    // GN_AGS
    // LD_MMA
    // GD_STOGO
    // LD_CCSAQ
    // LD_SLSQP
    // LD_VAR2
    nlopt::opt fopt = nlopt::opt(nlopt::LN_COBYLA, fop->N * 3);
    fopt.set_param("inner_maxeval", 200);
    fopt.set_maxeval(200);
    std::vector<double> lb ; lb.resize(fop->N * 3, -2);
    
    std::vector<double> ub ; ub.resize(fop->N * 3, 2);
    fopt.set_lower_bounds(lb);
    fopt.set_upper_bounds(ub);
    fopt.set_min_objective(opt_func, fop);
    double tol = 1e-6;
    // std::vector<double> tols {1e-3, 1e-3, 1e-3, 1e-3};
    // fopt.add_inequality_constraint(constraint, fop, tol);
    fopt.set_xtol_rel(tol);
    fopt.set_force_stop(tol);
    return fopt;
}