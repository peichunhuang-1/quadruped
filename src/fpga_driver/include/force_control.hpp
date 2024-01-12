#include <Eigen/Dense>
#include <iostream>
#include <linkleg_kinematics.hpp>

class ForceTracker
{
public:
    ForceTracker();
    // Track desired force F_d and reference trajectory X_r
    void track(double F_d, Eigen::Vector3d X_r);
    double M_d; // desired inertia
    double K_d; // desired stiffness
    double B_d; // desired damping
};