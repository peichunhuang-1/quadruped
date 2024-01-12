#ifndef OPT_HPP
#define OPT_HPP
#include "Eigen/Dense"
#include "nlopt.hpp"
#include "ObservationModel.hpp"

class VelocityOptimizer {
    public:
        VelocityOptimizer() {
            states.resize(4);
        }
        std::vector<STATE> states;
        
};

double opt_func(const std::vector<double> &x, std::vector<double> &grad, void *f_data){
    if (grad.size() == 0)grad.resize(3);
    VelocityOptimizer *fop = (VelocityOptimizer*) (f_data);
    Eigen::Vector3d v = Eigen::Vector3d(x.data());
    Eigen::Vector3d gradient = Eigen::Vector3d(0, 0, 0);
    double cost = 0;
    double sum = 1e-10;
    for (int i = 0; i < 4; i ++) sum += fop->states[i].weight;
    
    for (int i = 0; i < 4; i ++) {
        Eigen::Vector3d diff = v - fop->states[i].predicted_velocity;
        double vpv = std::sqrt(diff.transpose() * fop->states[i].covariance.inverse() * diff);
        cost += fop->states[i].weight / sum * vpv;
        if (vpv != 0) {
            gradient += 0.5 * fop->states[i].weight / vpv * ((diff.transpose() * fop->states[i].covariance.inverse()).transpose() + fop->states[i].covariance.inverse() * diff);
        }
    }
    
    grad[0] = gradient(0);
    grad[1] = gradient(1);
    grad[2] = gradient(2);
    return cost;
}

nlopt::opt Optimizer(VelocityOptimizer* fop)
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
    nlopt::opt fopt = nlopt::opt(nlopt::LD_MMA, 3);
    fopt.set_param("inner_maxeval", 30);
    fopt.set_maxeval(30);
    std::vector<double> lb {-2, -2, -2};
    
    std::vector<double> ub {2, 2, 2};

    fopt.set_lower_bounds(lb);
    fopt.set_upper_bounds(ub);
    fopt.set_min_objective(opt_func, fop);
    double tol = 1e-4;
    // std::vector<double> tols {1e-3, 1e-3, 1e-3, 1e-3};

    // fopt.add_inequality_mconstraint(constraints, fop, tols);
    fopt.set_xtol_rel(tol);
    fopt.set_force_stop(tol);
    return fopt;
}

#endif