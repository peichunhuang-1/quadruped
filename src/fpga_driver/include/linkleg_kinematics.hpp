#include <Eigen/Dense>
#include <angle_convert.hpp>
#include <vector>
#include <assert.h>

/* Polynomail Coefficient by Curve fitting approx. */
std::vector<double> fk_coeff = {-0.0004362, 0.00715474, -0.04529168, 0.13936602, -0.22479562, 0.19297458, 0.01846436, 0.08280706};
std::vector<double> ik_coeff = {4.10837754 * pow(10, 5), -6.25071622 * pow(10, 5), 4.02616603 * pow(10, 5), -1.41754291 * pow(10, 5),
                                2.93469460 * pow(10, 4), -3.56228405 * pow(10, 3), 2.44240485 * pow(10, 2), -7.12383946};

std::vector<double> Polyval(std::vector<double> coeffs, std::vector<double> values)
{
    assert(coeffs.size() > 0);
    std::vector<double> results;
    for (auto const &val : values)
    {
        double result = coeffs[0];
        for (int i = 1; i < coeffs.size(); i++)
        {
            result *= val;
            result += coeffs[i];
        }
        results.push_back(result);
    }
    return results;
}

Eigen::Vector2d getPhiRL(Eigen::Vector2d v_tb)
{
    Eigen::Vector2d tb_ = v_tb;
    Eigen::Vector2d phi_;
    if (v_tb.coeff(1, 0) < -M_PI)
        tb_(1, 0) += 2 * M_PI;
    else
        tb_(1, 0) -= 2 * M_PI;

    Eigen::MatrixXd t(2, 2);
    Eigen::MatrixXd b(2, 1);
    t << 1, 1, -1, 1;
    b << deg2rad(17), -1 * deg2rad(17);
    phi_ = t * v_tb - b;
    return phi_;
}

Eigen::Vector2d getThetaBeta(Eigen::Vector2d v_phiRL)
{
    Eigen::Vector2d tb_;
    double phiR = v_phiRL.coeff(0, 0);
    double phiL = v_phiRL.coeff(1, 0);
    std::complex<double> compR(0, phiR + deg2rad(17.0));
    std::complex<double> compL(0, phiL - deg2rad(17.0));
    std::complex<double> delta = std::exp(compR) / std::exp(compL);
    double theta = std::arg(delta) / 2;
    double beta = std::arg(std::exp(compL)) + theta;
    tb_ << theta, beta;
    // std::cout << "phiR, phiL = " << phiR << " " << phiL << std::endl;
    // std::cout << "theta, beta = " << theta << " " << beta << std::endl;
    return tb_;
}

Eigen::Vector2d fowardKinematics(Eigen::Vector2d v_joint, std::string frame)
{
    Eigen::Vector2d tb;
    if (frame == "phi")
        tb = getThetaBeta(v_joint);
    else
        tb = v_joint;

    double theta = tb.coeff(0, 0);
    double beta = tb.coeff(1, 0);
    // std::cout << "theta, beta = " << theta << " " << beta << std::endl;
    std::vector<double> val = {theta};
    double length = Polyval(fk_coeff, val)[0];
    double footend_x = length * -sin(beta);
    double footend_y = length * -cos(beta);
    Eigen::Vector2d footend;
    footend << footend_x, footend_y;
    return footend;
}

Eigen::Vector2d inverseKinematics(Eigen::Vector2d v_footend, std::string frame)
{
    Eigen::Vector2d v_joint;

    double length = v_footend.norm();
    std::vector<double> val = {length};
    double theta = Polyval(ik_coeff, val)[0];

    Eigen::Vector2d u_footend = v_footend / v_footend.norm();
    double beta = acos(-u_footend(1, 0));
    if (v_footend(0, 0) > 0)
        beta = fabs(beta) * -1;
    else
        beta = fabs(beta);

    v_joint << theta, beta;

    if (frame == "phi")
    {
        v_joint = getPhiRL(v_joint);
    }
    return v_joint;
}

Eigen::Matrix2d footendJacobian(Eigen::Vector2d v_joint, std::string frame, double step_size)
{
    /* Obtain footend Jacobian matrix by approximation */
    Eigen::Matrix2d Jacobian;
    double x1 = v_joint(0, 0);
    double x2 = v_joint(1, 0);
    double x1_p = x1 + step_size;
    double x1_n = x1 - step_size;
    double x2_p = x2 + step_size;
    double x2_n = x2 - step_size;
    Eigen::Vector2d v11;
    v11 << x1_p, x2;
    Eigen::Vector2d v12;
    v12 << x1_n, x2;
    Eigen::Vector2d v21;
    v21 << x1, x2_p;
    Eigen::Vector2d v22;
    v22 << x1, x2_n;

    Eigen::Vector2d d_x1 = (fowardKinematics(v11, frame) - fowardKinematics(v12, frame)) / (2 * step_size);
    Eigen::Vector2d d_x2 = (fowardKinematics(v21, frame) - fowardKinematics(v22, frame)) / (2 * step_size);

    Jacobian << d_x1(0, 0), d_x2(0, 0), d_x1(1, 0), d_x2(1, 0);

    return Jacobian;
}

double deg2rad(double deg)
{
    return (deg / 180.0) * M_PI;
}

double rad2deg(double rad)
{
    return (rad / M_PI) * 180.0;
}
