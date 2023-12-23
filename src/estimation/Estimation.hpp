#include "ContactMap.hpp"

Eigen::Vector3d velocity(Eigen::Matrix<double, Eigen::Dynamic, 4> encoder_data, // theta, beta, theta_d, beta_d
                        Eigen::Matrix<double, Eigen::Dynamic, 3> omega,
                        Eigen::Matrix<double, Eigen::Dynamic, 4> quaternion,
                        Eigen::Matrix<double, Eigen::Dynamic, 3> acceleration, double dt, Leg &leg, double alpha_0 = 0, size_t N = 1) {
    Eigen::Vector3d rolling(0, 0, 0);
    Eigen::Vector3d accelerating(0, 0, 0);
    encoder_data.resize(N, 4);
    omega.resize(N, 3);
    quaternion.resize(N, 4);
    acceleration.resize(N, 3);
    ContactMap cm;
    double contact_beta = encoder_data.row(0)(1) + alpha_0;
    for (int i = 0; i < N; i++) {
        Eigen::Vector4d encoder_k = encoder_data.row(i);
        Eigen::Vector4d q = quaternion.row(i);
        Eigen::Quaterniond quat = Eigen::Quaterniond(q);
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        contact_beta += (encoder_k(3) + omega.row(i)(1) - 1e-3) * dt;
        leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
        accelerating += dt * dt * 0.5 * (2*i + 1) * (rot * (acceleration.row(i) - Eigen::Vector3d(1e-2, 1e-2, 1e-2).transpose()).transpose()).transpose();
        rolling += (dt * rot * leg.RollVelocity(omega.row(i) - Eigen::Vector3d(1e-3, 1e-3, 1e-3).transpose(), cm.lookup(encoder_k(0), contact_beta), 0)).transpose();
    }
    Eigen::Vector4d encoder_k = encoder_data.row(N-1);
    leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
    leg.PointContact(cm.lookup(encoder_k(0), encoder_k(1)), 0);
    Eigen::Vector4d q = quaternion.row(N-1);
    Eigen::Quaterniond quat = Eigen::Quaterniond(q);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Vector3d pose_k = rot * leg.contact_point;
    
    encoder_k = encoder_data.row(0);
    leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
    leg.PointContact(cm.lookup(encoder_k(0), contact_beta), 0);
    q = quaternion.row(0);
    quat = Eigen::Quaterniond(q);
    rot = quat.toRotationMatrix();
    Eigen::Vector3d pose_kn_last = rot * leg.contact_point;
    return (pose_kn_last - pose_k - rolling - accelerating) / ((double) N * dt);
}

Eigen::Vector3d velocity2(Eigen::Vector4d encoder_data, // theta, beta, theta_d, beta_d
                        Eigen::Vector3d omega,
                        Eigen::Vector4d quaternion,
                        Leg &leg) {
    ContactMap cm;
    leg.Calculate(encoder_data(0), encoder_data(2), 0, encoder_data(1), encoder_data(3), 0);
    leg.PointVelocity(Eigen::Vector3d(0, 0, 0), omega, cm.lookup(encoder_data(0), encoder_data(1)), 0);
    Eigen::Quaterniond quat = Eigen::Quaterniond(quaternion);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    Eigen::Vector3d vel_k = - rot * leg.contact_velocity;
    return vel_k;
}

