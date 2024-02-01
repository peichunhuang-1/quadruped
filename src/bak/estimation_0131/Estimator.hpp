#ifndef OBSERVATIONMODEL_HPP
#define OBSERVATIONMODEL_HPP
#include <vector>
#include <Eigen/Dense>
#include "nlopt.hpp"
#include "ContactMap.hpp"
#include <deque>

struct IMU_DATA {
    Eigen::Vector3d a;
    Eigen::Vector3d w;
    Eigen::Vector4d q;
};

struct ENCODER_DATA {
    double beta;
    double theta;
    double beta_d;
    double theta_d;
};

struct LIDAR_DATA {
    double distance;
    double roll;
    double pitch;
};

struct GND_DATA {
    Eigen::Vector3d point;
    Eigen::Matrix3d rotation;
};

struct STATE {
    Eigen::Vector3d predicted_velocity;
    Eigen::Vector3d contact_point;
    Eigen::Matrix3d velocity_covariance;
    Eigen::Matrix3d point_covariance;
};

double gaussian_erf(double x) ; 

class VelocityEstimateByLeg{
    public:
    VelocityEstimateByLeg(Eigen::Vector3d offset, Eigen::Vector3d lidar_offset, double R, double r, double dt, int n) ;
    void calculate(IMU_DATA imu, ENCODER_DATA m, LIDAR_DATA lidar, bool update_) ;
    Leg leg ;
    Eigen::Vector3d lidar ;
    double durations ;
    IMU_DATA current_imu ;
    ENCODER_DATA current_m ;
    GND_DATA current_gnd ;
    LIDAR_DATA current_lidar ;
    std::deque<STATE> states;
    bool update;
    double contact_beta;
};

class BodyEstimation {
    public:
    BodyEstimation(VelocityEstimateByLeg *lf, VelocityEstimateByLeg *rf, VelocityEstimateByLeg *rh, VelocityEstimateByLeg *lh, int n) ;
    std::vector<VelocityEstimateByLeg *> legs ; 
    std::deque<Eigen::Vector3d> position ;
    std::deque<Eigen::Vector3d> velocity ;
    std::deque<Eigen::Matrix3d> rotation ;
    std::deque<Eigen::Vector4d> weights ;
    void calculate_ground(int index) ;
    void calculate_weight() ;
    void assign(std::vector<double> x) ;
    double sigma ;
    int N;
    void prediction(Eigen::Matrix3d rot) ;
};

double opt_func(const std::vector<double> &x, std::vector<double> &grad, void *f_data) ;

double contact_constraint(const std::vector<double> &x, std::vector<double> &grad, void *data) ;

nlopt::opt Optimizer(BodyEstimation* fop) ;

#endif