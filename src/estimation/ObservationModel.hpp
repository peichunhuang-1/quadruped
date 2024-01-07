#include <vector>
#include <Eigen/Dense>
#include "ContactMap.hpp"
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

struct STATE {
    Eigen::Vector3d predicted_velocity;
    Eigen::Vector3d observed_velocity;
    double contact_beta;
    Eigen::Matrix3d covariance;
    double weight;
};

class LegVelocityEstimation {
    public:
    LegVelocityEstimation(Eigen::Vector3d offset, double R, double r, double dt) : leg(offset, R, r), durations(dt) {
        double theta_cov = 1e-4;
        double beta_cov = 1e-4;
        double alpha_cov = 2.5e-3;
        double roll_cov = 2.5e-5;
        double pitch_cov = 2.5e-5;
        double yaw_cov = 4e-4;
        double omega_cov = 9e-4;
        double ax_cov = 9e-4;
        double ay_cov = 9e-4;
        double az_cov = 9e-4;
        double theta_d_cov = 1e-2;
        double beta_d_cov = 1e-2;

        covariance_1 << theta_cov, 0, 0, 0, 0, 0,
                        0, beta_cov, 0, 0, 0, 0,
                        0, 0, alpha_cov, 0, 0, 0,
                        0, 0, 0, roll_cov, 0, 0,
                        0, 0, 0, 0, pitch_cov, 0,
                        0, 0, 0, 0, 0, yaw_cov;
        covariance_2 << theta_cov, 0, 0, 0, 0, 0, 0, 0,
                        0, alpha_cov, 0, 0, 0, 0, 0, 0,
                        0, 0, roll_cov, 0, 0, 0, 0, 0,
                        0, 0, 0, pitch_cov, 0, 0, 0, 0,
                        0, 0, 0, 0, yaw_cov, 0, 0, 0,
                        0, 0, 0, 0, 0, theta_d_cov, 0, 0,
                        0, 0, 0, 0, 0, 0, beta_d_cov, 0,
                        0, 0, 0, 0, 0, 0, 0, omega_cov;
        covariance_3 << roll_cov, 0, 0, 0, 0, 0,
                        0, pitch_cov, 0, 0, 0, 0,
                        0, 0, yaw_cov, 0, 0, 0,
                        0, 0, 0, ax_cov, 0, 0,
                        0, 0, 0, 0, ay_cov, 0,
                        0, 0, 0, 0, 0, az_cov;
    }
    void imu_input(IMU_DATA imu) {
        imus.push_back(imu);
    }
    void encoder_input(ENCODER_DATA m) {
        encoders.push_back(m);
    }
    STATE observation(IMU_DATA imu, ENCODER_DATA m, DST_DATA d, bool update = false) {
        if (update) {
            std::vector<STATE>().swap(states);
            STATE state = {
                n_last_velocity(imu, m, d), 
                current_velocity(imu, m, m.beta + d.alpha),
                m.beta + d.alpha,
                fusing_ratio * Covariance(imu, m, d.alpha),
                weight(m, d)
            };
            Eigen::Vector3d in_corresponse_cov = (state.predicted_velocity - state.observed_velocity).array().square();
            state.covariance += (1 - fusing_ratio) * in_corresponse_cov.asDiagonal();
            states.push_back(state);
            std::vector<IMU_DATA>().swap(imus);
            std::vector<ENCODER_DATA>().swap(encoders);
            encoder_input(m);
            dst = d;
        }
        else {
            encoder_input(m);
            Eigen::Quaterniond quat = Eigen::Quaterniond(imus.back().q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            Eigen::Vector3d euler = rot.eulerAngles(0, 1, 2);
            double r = euler(0); double p = euler(1); double y = euler(2);
            Eigen::Matrix3d p_r1, p_r2, p_r3;
            Partial_RotationMatrix(r, p, y, p_r1, p_r2, p_r3);
            // calculate dv & covariance
            Eigen::Vector3d dv = durations * rot * (imus.back().a);
            Eigen::Vector3d j_acceleration_dr1 = durations * p_r1 * imus.back().a;
            Eigen::Vector3d j_acceleration_dr2 = durations * p_r2 * imus.back().a;
            Eigen::Vector3d j_acceleration_dr3 = durations * p_r3 * imus.back().a;
            Eigen::Matrix3d j_da = durations * rot;
            Eigen::Vector3d j_acceleration_dx = j_da.row(0);
            Eigen::Vector3d j_acceleration_dy = j_da.row(1);
            Eigen::Vector3d j_acceleration_dz = j_da.row(2);
            Eigen::Matrix<double, 6, 3> j3;
            j3.row(0) = j_acceleration_dr1;
            j3.row(1) = j_acceleration_dr2;
            j3.row(2) = j_acceleration_dr3;
            j3.row(3) = j_acceleration_dx;
            j3.row(4) = j_acceleration_dy;
            j3.row(5) = j_acceleration_dz;

            double contact_beta = states.back().contact_beta + durations * (m.beta_d + imu.w(1));
            STATE state = {
                states.back().predicted_velocity + dv, 
                current_velocity(imu, m, contact_beta),
                contact_beta,
                states.back().covariance + fusing_ratio * j3.transpose() * covariance_3 * j3,
                states.back().weight
            };
            Eigen::Vector3d in_corresponse_cov = (state.predicted_velocity - state.observed_velocity).array().square();
            state.covariance += (1 - fusing_ratio) * in_corresponse_cov.asDiagonal();
            states.push_back(state);
        }
        imu_input(imu);
    }
    STATE current() {return states.back();}

    private:
    Leg leg;
    Eigen::Matrix<double, 6, 6> covariance_1;
    Eigen::Matrix<double, 8, 8> covariance_2;
    Eigen::Matrix<double, 6, 6> covariance_3;
    double fusing_ratio = 0.8;
    double durations;
    std::vector<IMU_DATA> imus;
    std::vector<ENCODER_DATA> encoders;
    std::vector<STATE> states;
    DST_DATA dst;
    double gaussian_erf(double x) {
        return (std::erf(x) + 1) * 0.5;
    }
    double weight(ENCODER_DATA m, DST_DATA d) {
        ContactMap cm;
        leg.Calculate(m.theta, 0, 0, m.beta, 0, 0);
        leg.PointContact(cm.lookup(m.theta, m.beta+d.alpha), d.alpha);
        double dst_to_ground = -leg.contact_point(2);
        double sigma = (0.01 + abs(leg.contact_point(0)) * 0.1);
        return gaussian_erf((dst_to_ground - dst.dist) / sqrt(2) / sigma / sigma);
    }
    Eigen::Vector3d current_velocity(IMU_DATA imu, ENCODER_DATA m, double contact_beta) {
        ContactMap cm;
        leg.Calculate(m.theta, m.theta_d, 0, m.beta, m.beta_d, 0);
        leg.PointVelocity(Eigen::Vector3d(0, 0, 0), imu.w, cm.lookup(m.theta, contact_beta), contact_beta - m.beta);
        Eigen::Quaterniond quat = Eigen::Quaterniond(imu.q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        Eigen::Vector3d vel_k = - rot * leg.contact_velocity;
        return vel_k;
    }

    void Partial_RotationMatrix(double r, double p, double y, Eigen::Matrix3d &p_r1, Eigen::Matrix3d &p_r2, Eigen::Matrix3d &p_r3) {
        p_r1 << 0, sin(y) * sin(r) + cos(y) * sin(p) * cos(r), sin(y) * cos(r) - cos(y) * sin(p) * sin(r), 
                0, -cos(y) * sin(r) + sin(y) * sin(p) * cos(r), -cos(y) * cos(r) - sin(y) * sin(p) * sin(r),
                0, cos(p) * cos(r), -cos(p) * sin(r);
        p_r2 << -cos(y) * sin(p), cos(y) * cos(p) * sin(r), cos(y) * cos(p) * cos(r), 
                -sin(y) * sin(p), sin(y) * cos(p) * sin(r), sin(y) * cos(p) * cos(r),
                -cos(p), -sin(p) * sin(r), -sin(p) * cos(r);
        p_r3 << -sin(y) * cos(p), -cos(y) * cos(r) - sin(y) * sin(p) * sin(r), cos(y) * sin(r) - sin(y) * sin(p) * cos(r), 
                cos(y) * cos(p), -sin(p) * cos(r) + cos(y) * sin(p) * sin(r), sin(y) * sin(r) + cos(y) * sin(p) * cos(r),
                0, 0, 0;
    }
    Eigen::Matrix3d Position_Covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) {
        ContactMap cm;
        Eigen::Vector4d q = imu.q;
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        Eigen::Vector3d euler = rot.eulerAngles(0, 1, 2);
        double r = euler(0); double p = euler(1); double y = euler(2);
        RIM rim = cm.lookup(m.theta, m.beta + alpha);
        leg.Calculate(m.theta, 1, 0, m.beta, 0, 0);
        leg.PointVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), rim, alpha);
        Eigen::Vector3d j_dtheta = rot * leg.contact_velocity; // 1
        leg.Calculate(m.theta, 0, 0, m.beta, 1, 0);
        leg.PointVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), rim, alpha);
        Eigen::Vector3d j_dbeta = rot * leg.contact_velocity; // 2
        double rim_radius = rim == G_POINT? leg.radius() : leg.radius() + leg.Radius();
        Eigen::Vector3d j_dalpha = rot * Eigen::Vector3d(rim_radius * cos(M_PI + alpha), 0, -rim_radius * sin(M_PI + alpha)); // 3
        leg.Calculate(m.theta, 0, 0, m.beta, 0, 0);
        leg.PointContact(rim, alpha);
        Eigen::Matrix3d p_r1, p_r2, p_r3;
        Partial_RotationMatrix(r, p, y, p_r1, p_r2, p_r3);
        Eigen::Vector3d j_dr1 = p_r1 * leg.contact_point; // 4
        Eigen::Vector3d j_dr2 = p_r2 * leg.contact_point; // 5
        Eigen::Vector3d j_dr3 = p_r3 * leg.contact_point; // 6
        Eigen::Matrix<double, 6, 3> j;
        j.row(0) = j_dtheta;
        j.row(1) = j_dbeta;
        j.row(2) = j_dalpha;
        j.row(3) = j_dr1;
        j.row(4) = j_dr2;
        j.row(5) = j_dr3;
        return j.transpose() * covariance_1 * j;
    }

    Eigen::Matrix3d Covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) {
        if (encoders.empty()) return Eigen::Matrix3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

        ContactMap cm;
        int length = imus.size();
        double contact_beta = encoders[0].beta + dst.alpha;
        Eigen::Vector3d j_rolling_dtheta(0, 0, 0); // 1
        Eigen::Vector3d j_rolling_dalpha(0, 0, 0); // 3
        Eigen::Vector3d j_rolling_dr1(0, 0, 0); // 4
        Eigen::Vector3d j_rolling_dr2(0, 0, 0); // 5
        Eigen::Vector3d j_rolling_dr3(0, 0, 0); // 6
        Eigen::Vector3d j_rolling_dtheta_d(0, 0, 0); // 7
        Eigen::Vector3d j_rolling_dbeta_d(0, 0, 0); // 8
        Eigen::Vector3d j_rolling_domegay(0, 0, 0); // 9
        

        Eigen::Vector3d j_acceleration_dr1(0, 0, 0); // 4
        Eigen::Vector3d j_acceleration_dr2(0, 0, 0); // 5
        Eigen::Vector3d j_acceleration_dr3(0, 0, 0); // 6
        Eigen::Vector3d j_acceleration_dx(0, 0, 0); // 10
        Eigen::Vector3d j_acceleration_dy(0, 0, 0); // 11
        Eigen::Vector3d j_acceleration_dz(0, 0, 0); // 12
        double ratio = length * durations;
        for (int i = 0; i < length; i++) {
            // preparation
            Eigen::Vector4d q = imus[i].q;
            Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            Eigen::Vector3d euler = rot.eulerAngles(0, 1, 2);
            double r = euler(0); double p = euler(1); double y = euler(2);
            Eigen::Matrix3d p_r1, p_r2, p_r3;
            Partial_RotationMatrix(r, p, y, p_r1, p_r2, p_r3);
            contact_beta += (encoders[i].beta_d + imus[i].w(1)) * durations;

            // calculate j rolling
            // dtheta
            leg.Calculate(encoders[i].theta, encoders[i].theta_d, 0, encoders[i].beta, 0, 0);
            leg.RollVelocity(Eigen::Vector3d(0, 0, 0), cm.lookup(encoders[i].theta, contact_beta));
            double dlink_w_dtheta = encoders[i].theta_d == 0? 0: leg.link_w_d / encoders[i].theta_d;
            j_rolling_dtheta = durations * rot * Eigen::Vector3d(0, dlink_w_dtheta, 0).cross(Eigen::Vector3d(leg.rim_p.imag(), 0, leg.rim_p.real())) / ratio;
            // dtheta_d
            leg.Calculate(encoders[i].theta, encoders[i].theta_d, 1, encoders[i].beta, 0, 0);
            leg.RollVelocity(Eigen::Vector3d(0, 0, 0), cm.lookup(encoders[i].theta, contact_beta));
            j_rolling_dtheta_d = durations * rot * Eigen::Vector3d(0, leg.link_w_d - dlink_w_dtheta * encoders[i].theta_d, 0).cross(Eigen::Vector3d(leg.rim_p.imag(), 0, leg.rim_p.real()))  / ratio;
            // dr
            leg.Calculate(encoders[i].theta, encoders[i].theta_d, 0, encoders[i].beta, encoders[i].beta_d, 0);
            Eigen::Vector3d rv = leg.RollVelocity(imus[i].w, cm.lookup(encoders[i].theta, contact_beta));
            j_rolling_dr1 = durations * p_r1 * rv / ratio;
            j_rolling_dr2 = durations * p_r2 * rv / ratio;
            j_rolling_dr3 = durations * p_r3 * rv / ratio;
            // dbeta_d & d_omega_y
            j_rolling_dbeta_d = durations * rot * Eigen::Vector3d(0, 1, 0).cross(Eigen::Vector3d(leg.rim_p.imag(), 0, leg.rim_p.real())) / ratio;
            j_rolling_domegay = durations * j_rolling_dbeta_d / ratio;
            // dalpha
            j_rolling_dalpha = durations * rot * rv.cross(Eigen::Vector3d(leg.rim_p.real(), 0, -leg.rim_p.imag())) / ratio;
            
            // calculating j accel
            // dr 
            j_acceleration_dr1 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r1 * imus[i].a / ratio;
            j_acceleration_dr2 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r2 * imus[i].a / ratio;
            j_acceleration_dr3 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r3 * imus[i].a / ratio;
            Eigen::Matrix3d j_da = (durations * durations * 0.5 * (2*i + 1) + durations) * rot / ratio;
            j_acceleration_dx = j_da.row(0);
            j_acceleration_dy = j_da.row(1);
            j_acceleration_dz = j_da.row(2);
            Eigen::Matrix<double, 8, 3> j2;
            j2.row(0) = j_rolling_dtheta;
            j2.row(1) = j_rolling_dalpha;
            j2.row(2) = j_rolling_dr1;
            j2.row(3) = j_rolling_dr2;
            j2.row(4) = j_rolling_dr3;
            j2.row(5) = j_rolling_dtheta_d;
            j2.row(6) = j_rolling_dbeta_d;
            j2.row(7) = j_rolling_domegay;
            Eigen::Matrix<double, 6, 3> j3;
            j3.row(0) = j_acceleration_dr1;
            j3.row(1) = j_acceleration_dr2;
            j3.row(2) = j_acceleration_dr3;
            j3.row(3) = j_acceleration_dx;
            j3.row(4) = j_acceleration_dy;
            j3.row(5) = j_acceleration_dz;
            covariance += j2.transpose() * covariance_2 * j2;
            covariance += j3.transpose() * covariance_3 * j3;
        }
        covariance += Position_Covariance(imu, m, alpha) / ratio;
        covariance += Position_Covariance(imus[0], encoders[0], dst.alpha) / ratio;
        
        return covariance;
    }
    
    Eigen::Vector3d n_last_velocity(IMU_DATA imu, ENCODER_DATA m, DST_DATA d) {
        if (encoders.empty()) return Eigen::Vector3d(0, 0, 0);
        int length = imus.size();
        Eigen::Vector3d rolling(0, 0, 0);
        Eigen::Vector3d accelerating(0, 0, 0); // acceleration of position term
        Eigen::Vector3d acceleration(0, 0, 0); // acceleration of velocity term
        ContactMap cm;
        double contact_beta = encoders[0].beta + dst.alpha;
        for (int i = 0; i < length; i++) {
            Eigen::Vector4d q = imus[i].q;
            Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            contact_beta += (encoders[i].beta_d + imus[i].w(1)) * durations;
            leg.Calculate(encoders[i].theta, encoders[i].theta_d, 0, encoders[i].beta, encoders[i].beta_d, 0);
            accelerating += durations * durations * 0.5 * (2*i + 1) * (rot * (imus[i].a));
            acceleration += durations * (rot * (imus[i].a));
            rolling += (durations * rot * leg.RollVelocity(imus[i].w, cm.lookup(encoders[i].theta, contact_beta), contact_beta - encoders[i].beta));
        }

        leg.Calculate(m.theta, m.theta_d, 0, m.beta, m.beta_d, 0);
        leg.PointContact(cm.lookup(m.theta, m.beta + d.alpha), d.alpha);
        Eigen::Vector4d q = imu.q;
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        Eigen::Vector3d pose_k = rot * leg.contact_point;
        
        leg.Calculate(encoders[0].theta, encoders[0].theta_d, 0, encoders[0].beta, encoders[0].beta_d, 0);
        leg.PointContact(cm.lookup(encoders[0].theta, encoders[0].beta + dst.alpha), dst.alpha);
        q = imus[0].q;
        quat = Eigen::Quaterniond(q);
        rot = quat.toRotationMatrix();
        Eigen::Vector3d pose_kn_last = rot * leg.contact_point;
        return (pose_kn_last - pose_k - rolling - accelerating) / ((double) length * durations) + acceleration;
    }
};