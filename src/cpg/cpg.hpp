#ifndef CPG_HPP
#define CPG_HPP
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "NodeHandler.h"
#include "robot.pb.h"
#include "geometry.pb.h"
#include "math.h"
namespace Eigen{
    typedef Eigen::Vector<bool, 4> Vector4b;
}

double mod2pi(double ang) {
    while (ang > 2. * M_PI) ang -= 2. * M_PI;
    while (ang < 0) ang += 2. * M_PI;
    return ang;
}

class Kuramoto_cpg
{
    private:
        Eigen::Matrix<bool, 9, 4> ban_list;
        Eigen::Matrix<double, 4, 4> potential;
        double couple_gain = 0.1; 
        double potential_gain = 0.1;
        double zeta = 0.1;
        Eigen::Vector4d _phase;
        Eigen::Vector4b _contact;
        double swing_duration_last = 0;
    public:
        Kuramoto_cpg(Eigen::Matrix<bool, 9, 4> _ban_list, Eigen::Matrix<double, 4, 4> _potential)
        :ban_list(_ban_list), potential(_potential) {}
        void operator() (double c_gain, double p_gain, double z) {couple_gain = c_gain; potential_gain = p_gain; zeta = z;}
        void phase(Eigen::Vector4d phase) {_phase = phase;}
        Eigen::Vector4d phase() {return _phase;}
        Eigen::Vector4b contact() {return _contact;}
        void phase_generator(robot_msg::GAIT type, double dt, double* f = nullptr);
};

void Kuramoto_cpg::phase_generator(robot_msg::GAIT type, double dt, double* f)
{
    Eigen::Vector4d phase_gap;
    double swing_duration;
    Eigen::Vector4d nf; nf << 1, 1, 1, 1;
    switch (type)
    {
    case robot_msg::WALK:
    {
        phase_gap << M_PI/2., -M_PI/2., 0., -M_PI;
        swing_duration = M_PI/2.;
        break;
    }
    case robot_msg::TROT:
    {
        phase_gap << 0, M_PI, 0., M_PI;
        swing_duration = M_PI;
        break;
    }
    case robot_msg::STOP:
    {
        swing_duration = 0.0;
        Eigen::Vector4d K = cos(_phase.array());
        _contact = (K.array() < 0).matrix();
        swing_duration_last = swing_duration;
        return;
        break;
    }
    case robot_msg::STANCE:
    {
        swing_duration = 0.0;
        Eigen::Vector4d K = cos(_phase.array());
        _contact = (K.array() < 0).matrix();
        swing_duration_last = swing_duration;
        return;
        break;
    }
    default:
        phase_gap << M_PI/2., -M_PI/2., 0., -M_PI;
        swing_duration = M_PI/2. - 0.2;
        break;
    }

    for (int i = 0; i < 4; i++) {
        if (_contact(i)) {
            _phase(i) = swing_duration / swing_duration_last * (_phase(i) - M_PI) + M_PI;
        }
        else {
            if (_phase(i) > M_PI + swing_duration_last * 0.5) {
                _phase(i) = _phase(i) = (2 * M_PI - swing_duration) / (2 * M_PI - swing_duration_last) * (_phase(i) - 2. * M_PI) + 2. * M_PI;
            }
            else {
                _phase(i) = _phase(i) = (2 * M_PI - swing_duration) / (2 * M_PI - swing_duration_last) * _phase(i);
            }
        }
    }

    Eigen::Vector4d K = (cos(_phase.array()) + cos(swing_duration / 2.0)).matrix();
    _contact = (K.array() < 0).matrix();
    Eigen::Vector4d n; n << 0, 0, 0, 0;
    Eigen::Vector4b cross; cross << 0, 0, 0, 0;
    Eigen::Matrix<double, 4, 4> phase_gap_desired = phase_gap.transpose()({0, 0, 0, 0}, Eigen::placeholders::all) - phase_gap(Eigen::placeholders::all, {0, 0, 0, 0});
    for (int i = 0; i < 9; i++) {
        cross = (ban_list({i}, Eigen::placeholders::all).array() != _contact.transpose().array()).matrix();
        if (cross.cast <int> ().sum() == 1)
        n = n + cross.cast <double> ();
    }
    Eigen::Vector4d A =  - (potential_gain * exp(- potential_gain * K.transpose().array()) * sin(_phase.transpose().array()) * n.transpose().array()).matrix().transpose();
    Eigen::Matrix<double, 4, 4> phase_gap_state = nf * _phase.transpose() - _phase * nf.transpose();
    double oscillator = zeta * M_PI * 2.00;
    Eigen::Vector4d synchronizer = couple_gain * (potential.array() * sin((phase_gap_state - phase_gap_desired).array())).matrix() * nf;
    Eigen::Vector4d dp = ((synchronizer + A*(synchronizer.transpose()*synchronizer)).array() + oscillator).matrix();
    _phase = _phase + (dp.array() * dt).matrix();

    if (f != nullptr) 
        for (int i = 0; i < 4; i++)
            _phase(i) = _phase(i) + f[i] * dt;

    for (int i = 0; i < 4; i++) 
        _phase(i) = mod2pi(_phase(i));
    
    swing_duration_last = swing_duration;
    return;
}

#endif