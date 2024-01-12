#include <force_control.hpp>

ForceTracker::ForceTracker()
{
    M_d = 0.1;
    K_d = 0.1;
    B_d = 1;
}

int main()
{
    /* Eigen::Vector3d v1;
    Eigen::MatrixXd M1(3, 3);

    v1 << 0, 1, 2;
    std::cout << v1 << std::endl;
    std::cout << "v1_size " << v1.rows() << v1.cols() << std::endl;

    M1 << 0, 1, 2, 3, 4, 5, 5, 7, 8;

    std::cout << M1 << std::endl;
    M1(0, 0) += 1;
    std::cout << M1 << std::endl;
    std::cout << M1.coeff(0, 0) << std::endl; */

    Eigen::Vector2d phiRL;
    phiRL << 0, 0;
    getThetaBeta(phiRL);
    std::cout << getPhiRL(phiRL) << std::endl;
    Eigen::Vector2d tb;
    tb << deg2rad(17), deg2rad(0);
    std::cout << fowardKinematics(phiRL, "phi") << std::endl;
    tb << deg2rad(160), deg2rad(0);
    std::cout << fowardKinematics(tb, "tb") << std::endl;

    Eigen::Vector2d footend;
    footend << 0, -0.342951;
    std::cout << "-- ik --" << std::endl;
    std::cout << inverseKinematics(footend, "tb") << std::endl;
    tb << 1, 0.5;
    std::cout << footendJacobian(tb, "phi", 0.001) << std::endl;
}