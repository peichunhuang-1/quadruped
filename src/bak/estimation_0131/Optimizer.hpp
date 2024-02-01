#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Eigen/Dense"
#include "nlopt.hpp"
#include "ObservationModel.hpp"
double opt_func(const std::vector<double> &x, std::vector<double> &grad, void *f_data){ // x: {vx, vy, vz, wlf, wrf, wrh, wlh}
    BodyEstimation *fop = (BodyEstimation*) (f_data);
    Eigen::Vector3d v = Eigen::Vector3d(x[0], x[1], x[2]);
    double cost = 0;
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    double weight_sum = 0;
    for (int i = 0; i < 4; i ++) {
        Eigen::Vector3d diff = v - fop->legs[i]->states.back().predicted_velocity;
        double vpv = std::sqrt(diff.transpose() * fop->legs[i]->states.back().covariance.inverse() * diff);
        double weight_p = abs((x[3+i] - fop->weights(i)) / (fop->sigma));
        cost += x[3+i] * vpv + weight_p;
        cov += x[3+i] * fop->legs[i]->states.back().covariance;
        weight_sum += x[3+i];
    }
    Eigen::Vector3d diff = v - fop->prediction_velocity;
    double vpv = weight_sum * std::sqrt(diff.transpose() * fop->prediction_velocity_cov.inverse() * diff);
    cost += vpv;
    fop->velocity = v;
    fop->velocity_cov = cov / weight_sum;
    return cost;
}

double constraint(const std::vector<double> &x, std::vector<double> &grad, void *data) {
    return 0.5 - (x[3] + x[4] + x[5] + x[6]);
}

nlopt::opt Optimizer(BodyEstimation* fop)
{
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
    nlopt::opt fopt = nlopt::opt(nlopt::LN_COBYLA, 7);
    fopt.set_param("inner_maxeval", 300);
    fopt.set_maxeval(300);
    std::vector<double> lb {-2, -2, -2, 0, 0, 0, 0};
    
    std::vector<double> ub {2, 2, 2, 1, 1, 1, 1};
    fopt.set_lower_bounds(lb);
    fopt.set_upper_bounds(ub);
    fopt.set_min_objective(opt_func, fop);
    double tol = 1e-8;
    // std::vector<double> tols {1e-3, 1e-3, 1e-3, 1e-3};
    fopt.add_inequality_constraint(constraint, fop, tol);
    fopt.set_xtol_rel(tol);
    fopt.set_force_stop(tol);
    return fopt;
}

#endif