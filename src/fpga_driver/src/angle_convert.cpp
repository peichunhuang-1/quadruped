#include "angle_convert.hpp"

void getThetaBeta(double *theta_beta, double phi_R, double phi_L)
{
    // double theta_beta[2];
    theta_beta[0] = 0.5 * phi_R - 0.5 * phi_L + 1.0 * deg2rad(17);
    theta_beta[1] = 0.5 * phi_R + 0.5 * phi_L;
}

void getPhiVector(double *phiRL, double theta, double beta)
{
    // double phiRL[2];
    phiRL[0] = theta + beta - deg2rad(17);
    phiRL[1] = -1 * theta + beta + deg2rad(17);
    // return phiRL;
}

double deg2rad(double deg)
{
    return (deg / 180.0) * M_PI;
}