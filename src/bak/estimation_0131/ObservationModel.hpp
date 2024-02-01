#ifndef OBSERVATIONMODEL_HPP
#define OBSERVATIONMODEL_HPP
#include <vector>
#include <Eigen/Dense>
#include "ContactMap.hpp"
void Partial_RotationMatrix(Eigen::Matrix3d rot, Eigen::Matrix3d &p_r1, Eigen::Matrix3d &p_r2, Eigen::Matrix3d &p_r3) ;

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

struct DST_DATA {
    double dist;
    double alpha;
};

struct GND_DATA {
    Eigen::Vector3d point;
    Eigen::Matrix3d rotation;
};

struct STATE {
    Eigen::Vector3d predicted_velocity;
    double contact_beta;
    Eigen::Matrix3d covariance;
};
class BodyEstimation;

class LegVelocityEstimation {
    public:
        LegVelocityEstimation (Eigen::Vector3d offset, Eigen::Vector3d lidar_offset, double R, double r, double dt) ;
        STATE calculate(IMU_DATA imu, ENCODER_DATA m, DST_DATA d, bool update = false) ;

        Leg leg;
        Eigen::Vector3d lidar;
        Eigen::Matrix<double, 6, 6> covariance_contact_point ;
        Eigen::Matrix<double, 8, 8> covariance_rolling_velocity ;
        Eigen::Matrix<double, 6, 6> covariance_imu_accelerating ;
        Eigen::Vector3d velocity(IMU_DATA imu, ENCODER_DATA m, DST_DATA d) ;
        Eigen::Matrix3d rolling_covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) ;
        Eigen::Matrix3d acclerating_covariance(IMU_DATA imu) ;
        Eigen::Matrix3d imu_covariance(IMU_DATA imu) ;
        Eigen::Matrix3d position_covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) ;
        Eigen::Matrix3d covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) ;
        void imu_input(IMU_DATA imu) ;
        void encoder_input(ENCODER_DATA m) ;
        std::vector<STATE> states ;
        double durations ;
        std::vector<IMU_DATA> imus ;
        std::vector<ENCODER_DATA> encoders ;
        DST_DATA dst ;
        GND_DATA g;
        GND_DATA ground_info(Eigen::Vector3d position) {
            Eigen::Vector4d quaternion = imus.back().q;
            Eigen::Quaterniond quat = Eigen::Quaterniond(quaternion);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            Eigen::Matrix3d gnd_rot;
            gnd_rot << cos(-dst.alpha), 0, sin(-dst.alpha), 0, 1, 0, -sin(-dst.alpha), 0, cos(-dst.alpha);
            GND_DATA gnd = {
                position + rot * (lidar + Eigen::Vector3d(0, 0, -dst.dist)), 
                rot * gnd_rot
            };
            return gnd;
        }
        double gaussian_erf(double x) {
            return (std::erf(x) + 1) * 0.5;
        }
        double weight(Eigen::Vector3d position, bool update, double sigma = 0.01) {
            ContactMap cm;
            Eigen::Quaterniond quat = Eigen::Quaterniond(imus.back().q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            leg.Calculate(encoders.back().theta, 0, 0, encoders.back().beta, 0, 0);
            leg.PointContact(cm.lookup(encoders.back().theta, states.back().contact_beta),  states.back().contact_beta - encoders.back().beta);
            Eigen::Vector3d point_of_contact = position + rot * leg.contact_point;
            if (update) g = ground_info(position);
            Eigen::Vector3d N = g.rotation.transpose().row(2);
            double dst = (-N.dot(point_of_contact) + N.dot(g.point)) / (N.norm());
            return gaussian_erf(dst / sqrt(2 * sigma * sigma));
        }
};

class BodyEstimation {
    // manipulate order
    // optimize a velocity and covariance
    // prediction
    // update_weight
    public:
        BodyEstimation(LegVelocityEstimation *lf, LegVelocityEstimation *rf, LegVelocityEstimation *rh, LegVelocityEstimation *lh, Eigen::Vector3d v_init) {
            current_position = Eigen::Vector3d(0, 0, 0.11);
            weights = Eigen::Vector4d(.5, .5, .5, .5);
            legs.push_back(lf);
            legs.push_back(rf);
            legs.push_back(rh);
            legs.push_back(lh);
            velocity = v_init;
            prediction_velocity = v_init;
            prediction_velocity_cov = Eigen::Matrix3d::Identity();
            velocity_cov = Eigen::Matrix3d::Identity();
            velocity_bias = Eigen::Vector3d(0, 0, 0);
        }
        void prediction() {
            Eigen::Quaterniond quat = Eigen::Quaterniond(legs[0]->imus.back().q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            current_position += velocity * legs[0]->durations + 0.5 * legs[0]->durations * legs[0]->durations * rot * legs[0]->imus.back().a;
            prediction_velocity = velocity + legs[0]->durations * rot * legs[0]->imus.back().a;
            prediction_velocity_cov = velocity_cov + legs[0]->durations * legs[0]->imu_covariance(legs[0]->imus.back());
        }
        void update_weight(bool update) {
            for (int i = 0; i < 4; i++) weights(i) = legs[i]->weight(current_position, update, sigma);
        }
        double sigma = 1e-1;
        std::vector<LegVelocityEstimation *> legs;
        Eigen::Vector4d weights;
        Eigen::Vector3d current_position;
        Eigen::Vector3d velocity;
        Eigen::Vector3d velocity_bias;
        Eigen::Matrix3d velocity_cov;
        Eigen::Vector3d prediction_velocity;
        Eigen::Matrix3d prediction_velocity_cov;
};

#endif