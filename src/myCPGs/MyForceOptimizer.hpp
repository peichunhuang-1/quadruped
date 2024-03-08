#ifndef OPT_HPP
#define OPT_HPP
#include "Eigen/Dense"
#include "kinematic/RigidBodyKinematic.hpp"
#include "nlopt.hpp"

namespace RigidBodyDynamic {

class Feature {
    public:
    double Fs = 0.1; // static coefficient;
    Eigen::Vector3d vector = Eigen::Vector3d(0, 0, 1);
};

Eigen::Matrix<double, 12, 3> Jacobian(RigidBody r, std::string joint, double duration)
{
    // jacobian calculate {r, v, phi, omega}
    Eigen::Matrix3d C = Eigen::Quaterniond(r.state.attitude).toRotationMatrix();
    Eigen::Matrix<double, 12, 3> j;
    j << duration * duration * 0.5 / r.mass * Eigen::Matrix3d::Identity() * C, duration / r.mass * Eigen::Matrix3d::Identity() * C
      , duration * duration * 0.5 * r.inertia.inverse() * Eigen::skew3(r.joints[joint].position) ,  duration * r.inertia.inverse() * Eigen::skew3(r.joints[joint].position) ;
    return j;
}

class ForceOptimizer
{
    public:
        ForceOptimizer() {}
        ForceOptimizer(Eigen::Vector<double, 12> p, Eigen::Vector<double, 12> q, std::vector<double> m);
        Eigen::Matrix<double, 12, 12> P;
        Eigen::Matrix<double, 12, 12> Q;
        std::vector<double> input_mask;
        Eigen::Matrix<double, 12, 12> J;
        Eigen::Vector3d gravity = {0, 0, -9.81};
        State ctrl;
        RigidBody rigid_body;
        std::vector<Feature> land_form;
        double duration;
        double constraints[8];
        bool contact[4] = {false, false, false, false}; // false = contact
        Eigen::Vector<double, 12> Diff(State l, State r) {
            Eigen::Vector3d d_p, d_v, d_phi, d_omega;
            Eigen::Vector<double, 12> delta;
            d_p = Eigen::Vector3d(0, 0, 0);
            d_v = l.velocity - r.velocity;
            d_omega = l.angular_velocity - r.angular_velocity;

            Eigen::Quaterniond q_r = Eigen::Quaterniond(r.attitude);
            Eigen::Quaterniond d_q = Eigen::Quaterniond(l.attitude) * q_r.inverse();
            d_phi = d_q.toRotationMatrix().eulerAngles(0, 1, 2);
            delta << d_p, d_v, d_phi, d_omega;
            return delta;
        }
};
ForceOptimizer::ForceOptimizer(Eigen::Vector<double, 12> p, Eigen::Vector<double, 12> q ,std::vector<double> mask)
{
    P = p.matrix().asDiagonal();
    Q = q.matrix().asDiagonal();
    input_mask = mask ;
}

double opt_func(const std::vector<double> &x, std::vector<double> &grad, void *f_data)
{
    ForceOptimizer *fop = (ForceOptimizer*) (f_data);
    RigidBody r = fop->rigid_body;

    Eigen::Matrix3d C = Eigen::Quaterniond(r.state.attitude).toRotationMatrix();
    std::vector<double> input(12);
    std::transform(x.begin(), x.end(), fop->input_mask.begin(), input.begin(), std::multiplies<double>() );
    std::vector<Eigen::Vector3d> forces(4);
    Eigen::Vector<double, 12> u;
    for (int i = 0; i < 4; i++) {
        if (fop->contact[i]) forces[i] = Eigen::Vector3d(0, 0, 0);
        else forces[i] = Eigen::Vector3d(input.data()+i*3);
    }
    u << forces[0], forces[1], forces[2], forces[3];
    r.apply_force("centroid", C.transpose() * fop->rigid_body.mass * fop->gravity);

    r.apply_force("lf", forces[0]);
    r.apply_force("rf", forces[1]);
    r.apply_force("rh", forces[2]);
    r.apply_force("lh", forces[3]);
    r.update(fop->duration);
    
    Eigen::Vector<double, 12> s_err = fop->Diff(fop->ctrl, r.state);
    printf("%3f\n", s_err.norm());
    Eigen::Vector<double, 12> gradient = -2 * (s_err.transpose() * fop->P * fop->J - u.transpose() * fop->Q);
    for (int i = 0; i < 12; i++) 
        grad[i] = gradient(i);
    for (int i = 0; i < 4; i++) {
        if (fop->contact[i]) {
            grad[i*3] = 0;
            grad[i*3+1] = 0;
            grad[i*3+2] = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d force = Eigen::Vector3d(input.data() + i * 3);
        force = C * force;
        double fs = fop->land_form[i].Fs;

        fop->constraints[i] = - force.dot(fop->land_form[i].vector);
        fop->constraints[i*2] = force.dot(force) - (1+fs*fs) * fop->constraints[i] * fop->constraints[i] - 1e-10;
        fop->constraints[i] -= 1e-10;
    }
    double result = s_err.transpose() * fop->P * s_err;
    result += u.transpose() * fop->Q * u;
    return result;
}

void constraints(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data)
{
    ForceOptimizer *fop = (ForceOptimizer*) (f_data);
    for ( int i = 0; i < m; i ++) {
        result[i] = fop->constraints[i];
    }
}


nlopt::opt Optimizer(ForceOptimizer* fop)
{
    nlopt::opt fopt = nlopt::opt(nlopt::LD_MMA, 12);
    fopt.set_param("inner_maxeval", 1000);
    fopt.set_maxeval(1000);
    std::vector<double> lb {-200, 0, -200, 
                            -200, 0, -200,
                            -200, 0, -200, 
                            -200, 0, -200};
    
    std::vector<double> ub {200, 0, 200, 
                            200, 0, 200,
                            200, 0, 200, 
                            200, 0, 200};

    fopt.set_lower_bounds(lb);
    fopt.set_upper_bounds(ub);
    fopt.set_min_objective(opt_func, fop);
    double tol = 1e-5;
    std::vector<double> tols {1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2};

    fopt.add_inequality_mconstraint(constraints, fop, tols);
    fopt.set_xtol_abs(tol);
    fopt.set_force_stop(tol);
    return fopt;
}
}

#endif