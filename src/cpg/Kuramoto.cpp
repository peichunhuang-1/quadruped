#include "Kuramoto.hpp"

namespace Kuramoto {
SynchronizeController::SynchronizeController() {}
SynchronizeController::SynchronizeController(double Kp, double Ki, double Kd, double max, double min, double windup, double winddown) {
    p = Kp;
    i = Ki;
    d = Kd;
    Max = max;
    Min = min;
    i_term = 0;
    last_err = 0;
    up = windup;
    down = winddown;
}
double SynchronizeController::output(double err, double dt) {
    i_term += dt * err;
    i_term = i_term > up? up : i_term;
    i_term = i_term < down? down : i_term;
    double out = p * err + d * (err - last_err) / dt + i * i_term;
    out = out > Max? Max: out;
    out = out < Min? Min: out;
    last_err = err;
    return out;
}
void SynchronizeController::clear() {
    i_term = 0;
    last_err = 0;
}

kuramoto_neuron::kuramoto_neuron(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_):
alpha(alpha_), beta(beta_), mu(mu_), omega_stance(omega_stance_), omega_swing(omega_swing_), b(b_)  {
    x = (double) (rand() / (RAND_MAX + 1.0)) * 1e-10;
    y = (double) (rand() / (RAND_MAX + 1.0)) * sqrt(mu);
    x_dot = - (double) (rand() / (RAND_MAX + 1.0)) *  1e-4;
    y_dot = - (double) (rand() / (RAND_MAX + 1.0)) *  1e-4;
    c = SynchronizeController(1.85, 0.01, 0.8);
}

void kuramoto_neuron::change_param(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_) {
    alpha = alpha_;
    beta = beta_;
    mu = mu_;
    omega_stance = omega_stance_;
    omega_swing = omega_swing_;
    b = b_;
}

void kuramoto_neuron::dxdt(double dv, TRANSITION_TYPE t, double dt) {
    if (!(FEEDBACK & t) ) {
        c.clear();
        dv = 0;
    }
    else {
        dv = c.output(dv, dt);
    }
    x_dot = alpha * (mu - radius * radius) * x - w * y - dv;
}

void kuramoto_neuron::dydt(double u, double Ky) {
    y_dot = beta * (mu - radius * radius) * y + w * x + u + Ky;
}

void kuramoto_neuron::r() {
    radius = sqrt(x * x + y * y);
}

void kuramoto_neuron::omega() {
    w = omega_stance / (exp(-b * y) + 1) + omega_swing / (exp(b * y) + 1);
}

void kuramoto_neuron::update(double Ky, double f, double dv, double dt, TRANSITION_TYPE t) {
    double u_ = u(f, Ky, t);
    r();
    omega();
    dxdt(dv, t, dt);
    dydt(u_, Ky);
    x += x_dot * dt;
    y += y_dot * dt;
}

double kuramoto_neuron::u(double f, double Ky, TRANSITION_TYPE t) {
    switch (t) {
        case FAST: {
            return y > 0? -f: f;
        } 
        break;
        case STOP: {
            return -w * x - Ky;
        }
        case FAST_FEEDBACK: {
            return y > 0? -f: f;
        } 
        break;
        case STOP_FEEDBACK: {
            return - w * x - Ky;
        }
        break;
        case DEFAULT: {
            return 0;
        }
        break;
        default:
            return 0;
        break;
    }
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