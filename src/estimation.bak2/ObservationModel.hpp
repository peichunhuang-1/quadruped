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
    Eigen::Vector3d predicted_position;
    Eigen::Vector3d predicted_velocity;
    double contact_beta;
    Eigen::Matrix3d covariance;
    GND_DATA ground;
};

class LegVelocityEstimation {
    public:
        LegVelocityEstimation (Eigen::Vector3d offset, Eigen::Vector3d lidar_offset, double R, double r, double dt) ;
        STATE calculate(IMU_DATA imu, ENCODER_DATA m, DST_DATA d, bool update = false) ;
    private:
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
        double normal_pdf(double x, double m, double s) {
            static const double inv_sqrt_2pi = 0.3989422804014327;
            double a = (x - m) / s;
            return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
        }
        double gaussian_erf(double x) {
            return (std::erf(x) + 1) * 0.5;
        }
        Eigen::Vector3d predict_position(STATE s, IMU_DATA imu) {
            Eigen::Quaterniond quat = Eigen::Quaterniond(imu.q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            return s.predicted_position + s.predicted_velocity * durations + 0.5 * durations * durations * rot * imu.a;
        }
        GND_DATA ground_info(Eigen::Vector3d position, Eigen::Vector4d quaternion, DST_DATA dst) {
            Eigen::Quaterniond quat = Eigen::Quaterniond(quaternion);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            Eigen::Matrix3d gnd_rot;
            gnd_rot << cos(-dst.alpha), 0, sin(-dst.alpha), 0, 1, 0, -sin(-dst.alpha), 0, cos(-dst.alpha);
            GND_DATA gnd = {
                position + rot * (lidar + Eigen::Vector3d(0, 0, -dst.dist)), 
                gnd_rot
            };
            return gnd;
        }

        double weight(STATE s, IMU_DATA imu, ENCODER_DATA m, GND_DATA g) {
            ContactMap cm;
            Eigen::Quaterniond quat = Eigen::Quaterniond(imu.q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            leg.Calculate(m.theta, 0, 0, m.beta, 0, 0);
            leg.PointContact(cm.lookup(m.theta, s.contact_beta), s.contact_beta - m.beta);
            Eigen::Vector3d point_of_contact = s.predicted_position + rot * leg.contact_point;
            Eigen::Vector3d N = g.rotation.transpose().row(0);
            double dst = (N.dot(point_of_contact) - N.dot(g.point)) / (N.norm());
            // return gaussian_erf(dst / sqrt(2 * 1e-4));
            return normal_pdf(dst, 0, 1e-2);
        }
};

#endif