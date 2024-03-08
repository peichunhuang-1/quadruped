#include "kinematic/RigidBodyKinematic.hpp"
#include <vector>
#include "MyForceOptimizer.hpp"
#include <iostream>
using namespace RigidBodyDynamic;
int main() {
    Eigen::Vector<double, 12> p {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Eigen::Vector<double, 12> q {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    q = 1e-5 * q;
    p = 1e6 * p;
    double mass = 20.0;
    double height = 0.15;
    double width = 0.3;
    double length = 0.5;
    Eigen::Matrix3d inertia; 
    inertia << mass*(height * height + width * width), 0, 0,
                0, mass*(height * height + length * length), 0,
                0, 0, mass*(width * width + length * length);
    State s;
    s.position = Eigen::Vector3d(0, 0, 0);
    s.velocity = Eigen::Vector3d(0, 0, 0);
    s.angular_velocity = Eigen::Vector3d(0, 0, 0);
    s.attitude = Eigen::Vector4d(0, 0, 0, 1);

    // optimizer data
    std::vector<double> mask {1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
    std::vector<Feature> landforms(4);
    ForceOptimizer fopt_p(p, q, mask);
    fopt_p.land_form = landforms;
    fopt_p.duration = 1e-3;
    fopt_p.rigid_body.mass = mass;
    fopt_p.rigid_body.inertia = inertia;
    fopt_p.rigid_body.state = s;
    fopt_p.ctrl = s;
    fopt_p.rigid_body.joints["centroid"] = Joint(0, 0, 0);
    fopt_p.rigid_body.joints["lf"] = Joint(0.2, 0.15, -0.1);
    fopt_p.rigid_body.joints["rf"] = Joint(0.2, -0.15, -0.1);
    fopt_p.rigid_body.joints["rh"] = Joint(-0.2, -0.15, -0.1);
    fopt_p.rigid_body.joints["lh"] = Joint(-0.2, 0.15, -0.1);
    nlopt::opt fopt = Optimizer(&fopt_p);
    std::vector<double> u {0, 0, 49, 0, 0, 0, 0, 0, 49, 0, 0, 0};
    double minf = 0;


    s.position = Eigen::Vector3d(0, 0, 0);
    s.velocity = Eigen::Vector3d(0, 0, 0);
    s.angular_velocity = Eigen::Vector3d(0, 0, 0);
    s.attitude = Eigen::Vector4d(0, 0, 0, 1);
    fopt_p.rigid_body.state = s;
    s.position = Eigen::Vector3d(0, 0, 0);
    s.velocity = Eigen::Vector3d(1e-3, 0, 0);
    s.angular_velocity = Eigen::Vector3d(0, 0, 0);
    s.attitude = Eigen::Vector4d(0, 0, 0, 1);
    fopt_p.ctrl = s;
    Eigen::Matrix<double, 12, 12> J;
    J << Jacobian(fopt_p.rigid_body, "lf", fopt_p.duration), 
        Jacobian(fopt_p.rigid_body, "rf", fopt_p.duration),
        Jacobian(fopt_p.rigid_body, "rh", fopt_p.duration),
        Jacobian(fopt_p.rigid_body, "lh", fopt_p.duration); 
    fopt_p.J = J;
    fopt_p.contact[0] = false;
    fopt_p.contact[1] = false;
    fopt_p.contact[2] = false;
    fopt_p.contact[3] = true;
    fopt.optimize(u, minf);
    Eigen::Vector<double, 12> U(u.data());
    std::cout << U.transpose() << "\n";
}