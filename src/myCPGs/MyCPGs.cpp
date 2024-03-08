#include "MyCPGs.hpp"

namespace Kuramoto {
double mu_func(double phi, double mu_max, double d) {
    return mu_max;
}


MCKsystem::MCKsystem(double M, double C, double K, double x0, double ulimit, double llimit) {
    m = M;
    c = C;
    k = K;
    zero_point = x0;
    x_dd = 0;
    x_d = 0;
    x = x0;
    u = ulimit;
    l = llimit;
}

void MCKsystem::F(double u, double dt) {
    x_dd = (u - c * x_d - k * (x - zero_point)) / m;
    x += x_d * dt + x_dd * 0.5 * dt * dt; 
    x_d += dt * x_dd;
    if (x > u) {
        x = u;
        x_d = 0;
        x_dd = k * (x - zero_point) / m;
    }
    if (x < l) {
        x = l;
        x_d = 0;
        x_dd = k * (x - zero_point) / m;
    }
}

kuramoto_neuron::kuramoto_neuron(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_):
alpha(alpha_), beta(beta_), mu(mu_), omega_stance(omega_stance_), omega_swing(omega_swing_), b(b_)  {
    x = (double) (rand() / (RAND_MAX + 1.0)) * 1e-10;
    y = (double) (rand() / (RAND_MAX + 1.0)) * sqrt(mu);
    x_dot = - (double) (rand() / (RAND_MAX + 1.0)) *  1e-4;
    y_dot = - (double) (rand() / (RAND_MAX + 1.0)) *  1e-4;
    beta_system = new MCKsystem(10, 20, 10, 0., M_PI_2 / 2., -M_PI_2 / 2.);
    stance_height_system = new MCKsystem(10, 20, 10, 0.2, 0.21, 0.17);
}

void kuramoto_neuron::change_param(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_) {
    alpha = alpha_;
    beta = beta_;
    mu = mu_;
    omega_stance = omega_stance_;
    omega_swing = omega_swing_;
    b = b_;
}

void kuramoto_neuron::dxdt() {
    x_dot = alpha * (mu_d * mu_d - radius * radius) * x - w * y;
}

void kuramoto_neuron::dydt(double Ky, double F) {
    type t = T(F);
    double u;
    if (t == DEFAULT) u = 0;
    else if (t == FAST) u = y > 0? -F: F;
    else u = - w * x - Ky;
    y_dot = beta * (mu_d * mu_d - radius * radius) * y + w * x + Ky + u;
}

std::pair<double, double> kuramoto_neuron::output(double beta_const, double lift, double dig, Leg &leg, double dt) {
    // v_k+1 and w_k+1 were from update force and velocity feedback
    // with these, we can calculate the origin output (theta beta) from CPGs, and the desired output (theta beta) from contact velocity
    // and turn the gap between these two, so we can get the desired virtual force on beta0 and stance0
    // finally get the new beta0 and stance 0, we can obtain the final output (theta beta)
    LinkLegModel linkleg(0.01, 0.1);
    double r = (stance_height_system->x + (lift / (exp(b * y) + 1) + dig / (exp(-b * y) + 1)) * y) / cos(beta_const * x + beta_system->x);
    double theta = linkleg.inverse(r, G_POINT);
    double beta = beta_const * x + beta_system->x;
    // {static const double ptheta_pr = 10.276788;
    // static const double pr_ptheta = 0.09730667;
    // double beta_d = beta_const * x_dot + beta_system->x_d;
    // // //  + y * y_dot * (lift * b * exp(b * y) / (exp(b * y) + 1) / (exp(b * y) + 1) - dig * b * exp(-b * y) / (exp(-b * y) + 1) / (exp(-b * y) + 1)) -> 0
    // double theta_d = ptheta_pr * ((stance_height_system->x_d + y_dot * (lift / (exp(b * y) + 1) + dig / (exp(-b * y) + 1))) / cos(beta) - beta_d * sin(beta) * (stance_height_system->x + (lift / (exp(b * y) + 1) + dig / (exp(-b * y) + 1)) * y) / cos(beta) / cos(beta));
    // leg.Calculate(theta, 0, 0, beta, 0, 0);
    // leg.PointContact(G_POINT, 0);
    // leg.PointVelocity(v, w, G_POINT, 0, true);
    // double s_dot_desired = -leg.contact_velocity(2);
    // // Eigen::Matrix3d rot_beta; 
    // // rot_beta << cos(-beta), 0, sin(-beta), 0, 1, 0, -sin(-beta), 0, cos(-beta);
    // // Eigen::Vector3d body_velocity_in_leg_coordinate = rot_beta * (-leg.contact_velocity);
    // // double vr = -body_velocity_in_leg_coordinate(2);
    // // double vb = -body_velocity_in_leg_coordinate(0);
    // // double beta_d_desired = -vb / r;
    // // double theta_d_desired = vr / pr_ptheta;
    // // double virtual_force = theta_d_desired - theta_d;
    // std::cout << s_dot_desired << "\t" << stance_height_system->x_d - y_dot * (lift / (exp(b * y) + 1) + dig / (exp(-b * y) + 1)) << "\n";
    // double virtual_force = s_dot_desired - stance_height_system->x_d - y_dot * (lift / (exp(b * y) + 1) + dig / (exp(-b * y) + 1)); 
    // // double virtual_torque = beta_d_desired - beta_d;
    // // std::cout << virtual_force << "\t" << virtual_torque << "\n";
    // // std::cout << beta_d_desired << "\t" << beta_d << "\t" << theta_d_desired << "\t" << theta_d << "\n";
    // if (apply_feedback) stance_height_system->F(virtual_force, dt);
    // // if (apply_feedback) beta_system->F(virtual_torque, dt);}
    return std::pair<double, double> (theta, beta);
}

kuramoto_neuron::type kuramoto_neuron::T(double F) {
    if (y > 0) {
        if (x < 0) {
            if (y < 0.01 * mu_d && x > -1.05 * mu_d && x < -0.95 * mu_d && F > 20.0) return DEFAULT;
            else return FAST;
        }
        else return DEFAULT;
    }
    else if (y < 0) {
        if (x > 0) {
            if (y > -0.01 * mu_d && x < 1.05 * mu_d && x > 0.95 * mu_d && F < 10.0) return DEFAULT;
            else return FAST;
        }
        else return DEFAULT;
    }
    else return DEFAULT;
}

void kuramoto_neuron::r() {
    radius = sqrt(x * x + y * y);
}

void kuramoto_neuron::omega() {
    w = (omega_stance / (exp(-b * y) + 1) + omega_swing / (exp(b * y) + 1));
    phi = atan2(y, x);
    mu_d = mu_func(phi, mu);
}

void kuramoto_neuron::update(double Ky, double F, double dt) {
    r();
    omega();
    dxdt();
    dydt(Ky, F);
    x += x_dot * dt;
    y += y_dot * dt;
}

const Eigen::Matrix4d trot_K() {
    return (Eigen::Matrix<double, 4, 4, Eigen::DontAlign>() << 
    0, -1, 1, -1,
    -1, 0, -1, 1,
    1, -1, 0, -1,
    -1, 1, -1, 0
    ).finished();
}
const Eigen::Matrix4d walk_K() {
    return (Eigen::Matrix<double, 4, 4, Eigen::DontAlign>() << 
    0, -1, -1, 1,
    -1, 0, 1, -1,
    1, -1, 0, -1,
    -1, 1, -1, 0
    ).finished();
}
const Eigen::Matrix4d pace_K() {
    return (Eigen::Matrix<double, 4, 4, Eigen::DontAlign>() << 
    0, -1, -1, 1,
    -1, 0, 1, -1,
    -1, 1, 0, -1,
    1, -1, -1, 0
    ).finished();
}
const Eigen::Matrix4d bound_K() {
    return (Eigen::Matrix<double, 4, 4, Eigen::DontAlign>() << 
    0, 1, -1, -1,
    1, 0, -1, -1,
    -1, -1, 0, 1,
    -1, -1, 1, 0
    ).finished();
}

}