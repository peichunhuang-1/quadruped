#ifndef LEG_HPP
#define LEG_HPP
#include <Eigen/Dense>
#include "LinkLegModel.hpp"


class Leg : public LinkLegModel
{
    public:
    Leg(Eigen::Vector3d offset, double R = 0.1, double r = 0.0125)
    : LinkLegModel(r, R)
    {
        this->offset = offset;
    }
    void Calculate(double theta, double theta_d, double theta_dd, double beta, double beta_d, double beta_dd);
    Eigen::Vector3d RimCentorVelocity(Eigen::Vector3d v, Eigen::Vector3d w, RIM rim);
    Eigen::Vector3d RimCentorPosition(RIM rim);
    void PointContact(RIM rim, double alpha=0);
    void PointVelocity(Eigen::Vector3d v, Eigen::Vector3d w, RIM rim, double alpha=0, bool inbody_coord=true);
    double RimRoll(RIM rim);
    Eigen::Vector3d RollVelocity(Eigen::Vector3d w, RIM rim, double alpha=0);
    double beta = 0;
    double beta_d = 0;
    double beta_dd = 0;
    Eigen::Vector3d offset;
    Eigen::Vector3d contact_point;
    Eigen::Vector3d contact_velocity;
    double Radius() {return this->R;}
    double radius() {return this->r;}
    std::pair<double, double> Inverse(Eigen::Vector3d p, RIM rim);
    double link_w = 0;
    double link_w_d = 0;
    std::complex<double> rim_p;

};
#endif