#include "ObservationModel.hpp"
LegVelocityEstimation::LegVelocityEstimation(Eigen::Vector3d offset, Eigen::Vector3d lidar_offset, double R, double r, double dt) : leg(offset, R, r), durations(dt), lidar(lidar_offset) {
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

    covariance_contact_point << theta_cov, 0, 0, 0, 0, 0,
                    0, beta_cov, 0, 0, 0, 0,
                    0, 0, alpha_cov, 0, 0, 0,
                    0, 0, 0, roll_cov, 0, 0,
                    0, 0, 0, 0, pitch_cov, 0,
                    0, 0, 0, 0, 0, yaw_cov;
    covariance_rolling_velocity << theta_cov, 0, 0, 0, 0, 0, 0, 0,
                    0, alpha_cov, 0, 0, 0, 0, 0, 0,
                    0, 0, roll_cov, 0, 0, 0, 0, 0,
                    0, 0, 0, pitch_cov, 0, 0, 0, 0,
                    0, 0, 0, 0, yaw_cov, 0, 0, 0,
                    0, 0, 0, 0, 0, theta_d_cov, 0, 0,
                    0, 0, 0, 0, 0, 0, beta_d_cov, 0,
                    0, 0, 0, 0, 0, 0, 0, omega_cov;
    covariance_imu_accelerating << roll_cov, 0, 0, 0, 0, 0,
                    0, pitch_cov, 0, 0, 0, 0,
                    0, 0, yaw_cov, 0, 0, 0,
                    0, 0, 0, ax_cov, 0, 0,
                    0, 0, 0, 0, ay_cov, 0,
                    0, 0, 0, 0, 0, az_cov;
    STATE state = {
        Eigen::Vector3d(0, 0, 0),
        0,
        1e-2 * Eigen::Matrix3d::Identity()
    };
    states.push_back(state);
}

Eigen::Matrix3d LegVelocityEstimation::rolling_covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) {
    if (encoders.empty()) return 1e-2 * Eigen::Matrix3d::Identity();
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
    double ratio = length * durations;
    for (int i = 0; i < length; i++) {
        Eigen::Vector4d q = imus[i].q;
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        Eigen::Matrix3d p_r1, p_r2, p_r3;
        Partial_RotationMatrix(rot, p_r1, p_r2, p_r3);
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
        Eigen::Matrix<double, 8, 3> j2;
        j2.row(0) = j_rolling_dtheta;
        j2.row(1) = j_rolling_dalpha;
        j2.row(2) = j_rolling_dr1;
        j2.row(3) = j_rolling_dr2;
        j2.row(4) = j_rolling_dr3;
        j2.row(5) = j_rolling_dtheta_d;
        j2.row(6) = j_rolling_dbeta_d;
        j2.row(7) = j_rolling_domegay;
        covariance += j2.transpose() * covariance_rolling_velocity * j2;  
    }
    return covariance;
}

Eigen::Matrix3d LegVelocityEstimation::acclerating_covariance(IMU_DATA imu) {
    if (imus.empty()) return 1e-2 * Eigen::Matrix3d::Identity();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    ContactMap cm;
    int length = imus.size();
    Eigen::Vector3d j_acceleration_dr1(0, 0, 0); // 4
    Eigen::Vector3d j_acceleration_dr2(0, 0, 0); // 5
    Eigen::Vector3d j_acceleration_dr3(0, 0, 0); // 6
    Eigen::Vector3d j_acceleration_dx(0, 0, 0); // 10
    Eigen::Vector3d j_acceleration_dy(0, 0, 0); // 11
    Eigen::Vector3d j_acceleration_dz(0, 0, 0); // 12
    double ratio = length * durations;
    for (int i = 0; i < length; i++) { 
        Eigen::Vector4d q = imus[i].q;
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        Eigen::Matrix3d p_r1, p_r2, p_r3;
        Partial_RotationMatrix(rot, p_r1, p_r2, p_r3);
        // dr 
        j_acceleration_dr1 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r1 * imus[i].a / ratio;
        j_acceleration_dr2 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r2 * imus[i].a / ratio;
        j_acceleration_dr3 = (durations * durations * 0.5 * (2*i + 1) + durations) * p_r3 * imus[i].a / ratio;
        Eigen::Matrix3d j_da = (durations * durations * 0.5 * (2*i + 1) + durations) * rot / ratio;
        j_acceleration_dx = j_da.row(0);
        j_acceleration_dy = j_da.row(1);
        j_acceleration_dz = j_da.row(2);
        Eigen::Matrix<double, 6, 3> j3;
        j3.row(0) = j_acceleration_dr1;
        j3.row(1) = j_acceleration_dr2;
        j3.row(2) = j_acceleration_dr3;
        j3.row(3) = j_acceleration_dx;
        j3.row(4) = j_acceleration_dy;
        j3.row(5) = j_acceleration_dz;
        covariance += j3.transpose() * covariance_imu_accelerating * j3;
    }
    return covariance;
}

Eigen::Matrix3d LegVelocityEstimation::imu_covariance(IMU_DATA imu) {
    Eigen::Quaterniond quat = Eigen::Quaterniond(imu.q);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Matrix3d p_r1, p_r2, p_r3;
    Partial_RotationMatrix(rot, p_r1, p_r2, p_r3);
    Eigen::Vector3d j_acceleration_dr1 = durations * p_r1 * imu.a;
    Eigen::Vector3d j_acceleration_dr2 = durations * p_r2 * imu.a;
    Eigen::Vector3d j_acceleration_dr3 = durations * p_r3 * imu.a;
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
    return j3.transpose() * covariance_imu_accelerating * j3;
}

Eigen::Matrix3d LegVelocityEstimation::position_covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) {
    ContactMap cm;
    Eigen::Vector4d q = imu.q;
    Eigen::Quaterniond quat = Eigen::Quaterniond(q);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    RIM rim = cm.lookup(m.theta, m.beta + alpha);
    leg.Calculate(m.theta, 1, 0, m.beta, 0, 0);
    leg.PointContact(cm.lookup(m.theta, rim), alpha);
    leg.PointVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), rim, alpha);
    Eigen::Vector3d j_dtheta = rot * leg.contact_velocity; // 1
    leg.Calculate(m.theta, 0, 0, m.beta, 1, 0);
    leg.PointContact(cm.lookup(m.theta, rim), alpha);
    leg.PointVelocity(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0), rim, alpha);
    Eigen::Vector3d j_dbeta = rot * leg.contact_velocity; // 2
    double rim_radius = rim == G_POINT? leg.radius() : leg.radius() + leg.Radius();
    Eigen::Vector3d j_dalpha = rot * Eigen::Vector3d(rim_radius * cos(M_PI + alpha), 0, -rim_radius * sin(M_PI + alpha)); // 3
    leg.Calculate(m.theta, 0, 0, m.beta, 0, 0);
    leg.PointContact(rim, alpha);
    Eigen::Matrix3d p_r1, p_r2, p_r3;
    Partial_RotationMatrix(rot, p_r1, p_r2, p_r3);
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
    return j.transpose() * covariance_contact_point * j;
}

Eigen::Vector3d LegVelocityEstimation::velocity(IMU_DATA imu, ENCODER_DATA m, DST_DATA d) {
    if (encoders.empty()) return Eigen::Vector3d(0, 0, 0);
    int length = imus.size();
    Eigen::Vector3d rolling(0, 0, 0);
    Eigen::Vector3d accelerating(0, 0, 0); // acceleration of position term
    Eigen::Vector3d acceleration(0, 0, 0); // acceleration of velocity term
    ContactMap cm;
    double contact_beta = encoders[0].beta + dst.alpha;
    RIM rim = cm.lookup(encoders[0].theta, contact_beta);
    Eigen::Vector3d last_contact_point;
    double last_theta = encoders[0].theta;
    double last_contact_beta = contact_beta;
    for (int i = 0; i < length; i++) {
        Eigen::Vector4d q = imus[i].q;
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        leg.Calculate(encoders[i].theta, encoders[i].theta_d, 0, encoders[i].beta, encoders[i].beta_d, 0);
        leg.PointContact(cm.lookup(encoders[i].theta, contact_beta), contact_beta - encoders[i].beta);
        accelerating += durations * durations * 0.5 * (2*i + 1) * (rot * (imus[i].a));
        acceleration += durations * (rot * (imus[i].a));
        rolling += (durations * rot * leg.RollVelocity(imus[i].w, cm.lookup(encoders[i].theta, contact_beta), contact_beta - encoders[i].beta));
        if (rim != cm.lookup(encoders[i].theta, contact_beta)) {
            rolling -= (rot * leg.contact_point - last_contact_point);
            if (rim == G_POINT) {
                if (cm.lookup(encoders[i].theta, contact_beta) == LOWER_RIM_R || cm.lookup(encoders[i].theta, contact_beta) == LOWER_RIM_L) {
                    double bound_beta = cm.boudary_beta(last_theta, last_contact_beta, encoders[i].theta, contact_beta);
                    double current_beta = contact_beta - bound_beta; cm.rad_mod2(current_beta);
                    double diff = (contact_beta - bound_beta); cm.rad_mod(diff);
                    rolling -= rot * Eigen::Vector3d((leg.Radius() - leg.radius()) * diff, 0, 0);
                }
            }
        }
        rim = cm.lookup(encoders[i].theta, contact_beta);
        last_theta = encoders[i].theta; 
        last_contact_beta = contact_beta;
        contact_beta += (encoders[i].beta_d + imus[i].w(1)) * durations;
        last_contact_point = rot * leg.contact_point;
    }

    leg.Calculate(m.theta, m.theta_d, 0, m.beta, m.beta_d, 0);
    RIM current_rim = cm.lookup(m.theta, m.beta + d.alpha);
    leg.PointContact(current_rim, d.alpha);
    Eigen::Vector4d q = imu.q;
    Eigen::Quaterniond quat = Eigen::Quaterniond(q);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Vector3d pose_k = rot * leg.contact_point;
    if (cm.lookup(m.theta, m.beta + d.alpha) != cm.lookup(encoders.back().theta, last_contact_beta)) {
        rolling -= (rot * leg.contact_point - last_contact_point);
        if (cm.lookup(encoders.back().theta, last_contact_beta) == G_POINT) {
            if (cm.lookup(m.theta, m.beta + d.alpha) == LOWER_RIM_R || cm.lookup(m.theta, m.beta + d.alpha) == LOWER_RIM_L) {
                double bound_beta = cm.boudary_beta(encoders.back().theta, last_contact_beta, m.theta, m.beta + d.alpha);
                double current_beta = m.beta + d.alpha; cm.rad_mod2(current_beta);
                double diff = (current_beta - bound_beta); cm.rad_mod(diff);
                rolling -= rot * Eigen::Vector3d( (leg.Radius() - leg.radius()) * diff, 0, 0);
            }
        }
    }
    leg.Calculate(encoders[0].theta, encoders[0].theta_d, 0, encoders[0].beta, encoders[0].beta_d, 0);
    RIM last_rim = cm.lookup(encoders[0].theta, encoders[0].beta + dst.alpha);
    leg.PointContact(last_rim, dst.alpha);
    q = imus[0].q;
    quat = Eigen::Quaterniond(q);
    rot = quat.toRotationMatrix();
    Eigen::Vector3d pose_kn_last = rot * leg.contact_point;
    Eigen::Vector3d differ_of_point = pose_kn_last - pose_k;
    return (differ_of_point - rolling - accelerating) / ((double) length * durations) + acceleration;
}


Eigen::Matrix3d LegVelocityEstimation::covariance(IMU_DATA imu, ENCODER_DATA m, double alpha) {
    if (encoders.empty()) return 1e-2 * Eigen::Matrix3d::Identity();
    int length = imus.size();
    double ratio = length * durations;
    Eigen::Matrix3d roll_cov = rolling_covariance(imu, m, alpha);
    Eigen::Matrix3d accel_cov = acclerating_covariance(imu);
    Eigen::Matrix3d pose_current_cov = position_covariance(imu, m, alpha) / ratio;
    Eigen::Matrix3d pose_last_current_cov = position_covariance(imus[0], encoders[0], dst.alpha) / ratio;
    return roll_cov + accel_cov + pose_current_cov + pose_last_current_cov;
}

void LegVelocityEstimation::imu_input(IMU_DATA imu) {
    imus.push_back(imu);
}

void LegVelocityEstimation::encoder_input(ENCODER_DATA m) {
    encoders.push_back(m);
}

STATE LegVelocityEstimation::calculate(IMU_DATA imu, ENCODER_DATA m, DST_DATA d, bool update) {
    STATE state;
    if (update) {
        STATE last_state;
        IMU_DATA last_imu;
        if (!states.empty()) last_state = states.back();
        if (!imus.empty()) last_imu = imus.back();
        else last_imu = imu;
        state = {
            velocity(imu, m, d), 
            m.beta + d.alpha,
            covariance(imu, m, d.alpha),
        };
        std::vector<STATE>().swap(states);
        states.push_back(state);
        std::vector<IMU_DATA>().swap(imus);
        std::vector<ENCODER_DATA>().swap(encoders);
        encoder_input(m);
        dst = d;
    }
    else {
        Eigen::Quaterniond quat = Eigen::Quaterniond(imus.back().q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        // calculate dv & covariance
        Eigen::Vector3d dv = durations * rot * (imus.back().a);
        Eigen::Matrix3d cov = imu_covariance(imus.back());
        double contact_beta = states.back().contact_beta + durations * (encoders.back().beta_d + imus.back().w(1));
        encoder_input(m);
        state = {
            states.back().predicted_velocity + dv, 
            contact_beta,
            states.back().covariance + durations * cov,
        };
        states.push_back(state);
    }
    imu_input(imu);
    return state;
}

void Partial_RotationMatrix(Eigen::Matrix3d rot, Eigen::Matrix3d &p_r1, Eigen::Matrix3d &p_r2, Eigen::Matrix3d &p_r3) {
    Eigen::Vector3d euler = rot.eulerAngles(0, 1, 2);
    double r = euler(0); double p = euler(1); double y = euler(2);
    p_r1 << 0, cos(r) * sin(p) * cos(y) + sin(r) * sin(y), -sin(r) * sin(p) * cos(y) + cos(r) * sin(y),
            0, cos(r) * sin(p) * sin(y) - sin(r) * cos(y), -sin(r) * sin(p) * sin(y) - cos(r) * cos(y),
            0, cos(r) * cos(p), -sin(r) * cos(p);
    p_r2 << -sin(p) * cos(y), sin(r) * cos(p) * cos(y), cos(r) * cos(p) * cos(y), 
            -sin(p) * sin(y), sin(r) * cos(p) * sin(y), cos(r) * cos(p) * sin(y),
            -cos(p), -sin(r) * sin(p), -cos(r) * sin(p);
    p_r3 << -cos(p) * sin(y), -sin(r) * sin(p) * sin(y) - cos(r) * cos(y), -cos(r) * sin(p) * sin(y) + sin(r) * cos(y),
            cos(p) * cos(y), sin(r) * sin(p) * cos(y) - cos(r) * sin(y), cos(r) * sin(p) * cos(y) + sin(r) * sin(y),
            0, 0, 0;
}