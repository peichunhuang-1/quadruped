#include "definition.hpp"

namespace estimation {

ground::ground() {}

ground::ground(Eigen::Vector3d norm, Eigen::Vector3d p) {
    this->orient = norm;
    this->position = p;
}

ground ground::coordinate(Eigen::Quaterniond q, Eigen::Vector3d p) {
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d new_orient = R * orient;
    Eigen::Vector3d new_position = p + R * position;
    return ground(new_orient, new_position);
}   

lidar::lidar(Eigen::Vector3d offset_) : offset(offset_) {}

Eigen::Vector3d lidar::measured_point(double dist) {
    if (dist > 0) measured = offset + Eigen::Vector3d(0, 0, -dist);
    return measured;
}

ground lidar::body_frame_ground (lidar a, lidar b) {
    Eigen::Vector3d v1 = a.measured_point() - measured;
    Eigen::Vector3d v2 = b.measured_point() - measured;
    Eigen::Vector3d v_normal = v1.cross(v2);
    if (v_normal(2) < 0) v_normal = -v_normal;
    v_normal = v_normal / v_normal.norm();
    return ground(v_normal, measured);
}

leg_states::leg_states(Leg &model) : leg(model) {}

void leg_states::calculate(double theta, double theta_d, double beta, double beta_d, Eigen::Vector3d w) {
    this->leg.Calculate(theta, theta_d, 0, beta, beta_d, 0) ;
    rim_center_points[UPPER_RIM_L] = this->leg.RimCentorPosition(UPPER_RIM_L);
    rim_center_points[LOWER_RIM_L] = this->leg.RimCentorPosition(LOWER_RIM_L);
    rim_center_points[UPPER_RIM_R] = this->leg.RimCentorPosition(UPPER_RIM_R);
    rim_center_points[LOWER_RIM_R] = this->leg.RimCentorPosition(LOWER_RIM_R);
    rim_center_points[G_POINT] = this->leg.RimCentorPosition(G_POINT);

    Eigen::Vector3d v0(0, 0, 0);
    rim_center_velocities[UPPER_RIM_L] = this->leg.RimCentorVelocity(v0, w, UPPER_RIM_L);
    rim_center_velocities[LOWER_RIM_L] = this->leg.RimCentorVelocity(v0, w, LOWER_RIM_L);
    rim_center_velocities[UPPER_RIM_R] = this->leg.RimCentorVelocity(v0, w, UPPER_RIM_R);
    rim_center_velocities[LOWER_RIM_R] = this->leg.RimCentorVelocity(v0, w, LOWER_RIM_R);
    rim_center_velocities[G_POINT] = this->leg.RimCentorVelocity(v0, w, G_POINT);

    rim_center_omega[UPPER_RIM_L] = this->leg.RimRoll(UPPER_RIM_L);
    rim_center_omega[LOWER_RIM_L] = this->leg.RimRoll(LOWER_RIM_L);
    rim_center_omega[UPPER_RIM_R] = this->leg.RimRoll(UPPER_RIM_R);
    rim_center_omega[LOWER_RIM_R] = this->leg.RimRoll(LOWER_RIM_R);
    rim_center_omega[G_POINT] = this->leg.RimRoll(G_POINT);
}

void leg_states::predict(Eigen::Quaterniond q, ground gnd, double dt) {
    double wheel_r = this->leg.Radius() + this->leg.radius();
    Eigen::Matrix3d R = q.toRotationMatrix();
    predicted_contact_points[UPPER_RIM_L] =  predicted_contact_points[UPPER_RIM_L] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[UPPER_RIM_L], 0)).cross(gnd.orient) ;
    predicted_contact_points[LOWER_RIM_L] = predicted_contact_points[LOWER_RIM_L] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[LOWER_RIM_L], 0)).cross(gnd.orient) ;
    predicted_contact_points[UPPER_RIM_R] = predicted_contact_points[UPPER_RIM_R] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[UPPER_RIM_R], 0)).cross(gnd.orient) ;
    predicted_contact_points[LOWER_RIM_R] = predicted_contact_points[LOWER_RIM_R] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[LOWER_RIM_R], 0)).cross(gnd.orient) ;
    predicted_contact_points[G_POINT] = predicted_contact_points[G_POINT] + dt * this->leg.radius() * (R * Eigen::Vector3d(0, rim_center_omega[G_POINT], 0)).cross(gnd.orient);
}

double leg_states::lookup_omega(RIM rim) {
    return rim_center_omega[rim];
}
Eigen::Vector3d leg_states::lookup_point(RIM rim) {
    return rim_center_points[rim];
}
Eigen::Vector3d leg_states::lookup_velocity(RIM rim) {
    return rim_center_velocities[rim];
}

Eigen::Vector3d leg_states::lookup_predicted_contact_point(RIM rim) {
    return predicted_contact_points[rim];
}

states::states(Eigen::Vector3d p, Eigen::Vector3d v, 
leg_states &lf_, leg_states &rf_, leg_states &rh_, leg_states &lh_, 
lidar &Llf_, lidar &Lrf_, lidar &Lrh_, lidar &Llh_) : p(p_), v(v_), 
lf(lf_), rf(rf_), rh(rh_), lh(lh_), 
Llf(Llf_), Lrf(Lrf_), Lrh(Lrh_), Llh(Llh_) {
    
}

void states::predict(Eigen::Vector3d a, Eigen::Quaterniond q, double dt) {
    Eigen::Matrix3d R = q.toRotationMatrix();
    p = p + v * dt + R * (a - ba) * dt * dt * 0.5;
    v = v + R * (a - ba) * dt;
    lf.predict(q, Glf, dt);
    rf.predict(q, Grf, dt);
    rh.predict(q, Grh, dt);
    lh.predict(q, Glh, dt);
}

void states::validate() {

}

}