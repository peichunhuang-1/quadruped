#include "definition.hpp"

namespace estimator {
double relu(double x, double mid_point, double max) { // integral should be 1
    // shape of function is a trapezoid
    double max_p = 2. / (2 * max - mid_point);
    return x > 0? x > max ? max_p : max_p / mid_point * x : 0;
}
double gaussianLikelihood(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
    int n = x.size();
    Eigen::VectorXd diff = x - mean;
    double exponent = -0.5 * diff.transpose() * covariance.inverse() * diff;
    double normalization = std::pow(2.0 * M_PI, -n / 2.0) * std::pow(covariance.determinant(), -0.5);
    double likelihood = normalization * std::exp(exponent);
    return likelihood;
}

ground::ground() {}

ground::ground(Eigen::Vector3d norm, Eigen::Vector3d p) {
    this->orient = norm;
    this->position = p;
}

ground ground::coordinate(Eigen::Matrix3d R, Eigen::Vector3d p) {
    // transform coordinate of ground measurement
    Eigen::Vector3d new_orient = R * orient;
    Eigen::Vector3d new_position = p + R * position;
    return ground(new_orient, new_position);
}   

void ground::predict(Eigen::Vector3d dR, double dz) {
    Eigen::Matrix3d m = (Eigen::AngleAxisd(dR(0), Eigen::Vector3d::UnitX())
    * Eigen::AngleAxisd(dR(1), Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(dR(2), Eigen::Vector3d::UnitZ())).toRotationMatrix();
    orient = m * orient;
    position = position + Eigen::Vector3d(0, 0, dz);
}

lidar::lidar(Eigen::Vector3d offset_) : offset(offset_) {}

Eigen::Vector3d lidar::measured_point(double dist) {
    if (dist > 0) measured = offset + Eigen::Vector3d(0, 0, -dist);
    return measured;
}

ground lidar::body_frame_ground (lidar a, lidar b) {
    // measure ground by lidar
    Eigen::Vector3d v1 = a.measured_point() - measured;
    Eigen::Vector3d v2 = b.measured_point() - measured;
    Eigen::Vector3d v_normal = v1.cross(v2);
    if (v_normal(2) < 0) v_normal = -v_normal;
    v_normal = v_normal / v_normal.norm();
    return ground(v_normal, measured);
}

leg_states::leg_states(Leg &model) : leg(model) {}

void leg_states::calculate(double theta, double theta_d, double beta, double beta_d, Eigen::Vector3d w) {
    // calculate leg states
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

void leg_states::predict(Eigen::Matrix3d R, ground gnd, double dt) {
    // predict leg contact when in contact phase
    double wheel_r = this->leg.Radius() + this->leg.radius();
    predicted_contact_points[UPPER_RIM_L] =  predicted_contact_points[UPPER_RIM_L] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[UPPER_RIM_L], 0)).cross(gnd.orient) ;
    predicted_contact_points[LOWER_RIM_L] = predicted_contact_points[LOWER_RIM_L] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[LOWER_RIM_L], 0)).cross(gnd.orient) ;
    predicted_contact_points[UPPER_RIM_R] = predicted_contact_points[UPPER_RIM_R] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[UPPER_RIM_R], 0)).cross(gnd.orient) ;
    predicted_contact_points[LOWER_RIM_R] = predicted_contact_points[LOWER_RIM_R] + dt * wheel_r * (R * Eigen::Vector3d(0, rim_center_omega[LOWER_RIM_R], 0)).cross(gnd.orient) ;
    predicted_contact_points[G_POINT] = predicted_contact_points[G_POINT] + dt * this->leg.radius() * (R * Eigen::Vector3d(0, rim_center_omega[G_POINT], 0)).cross(gnd.orient);
}

void leg_states::assign_contact_point(Eigen::Vector3d p, Eigen::Matrix3d R, RIM rim) {
    // assign leg states when in swing phase
    predicted_contact_points[rim] = p + R * rim_center_points[rim];
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

states::states(Eigen::Vector3d p_, Eigen::Vector3d v_, 
leg_states &lf_, leg_states &rf_, leg_states &rh_, leg_states &lh_, 
lidar &Llf_, lidar &Lrf_, lidar &Lrh_, lidar &Llh_,
ground &Glf_, ground &Grf_, ground &Grh_, ground &Glh_) : p(p_), v(v_), 
lf(lf_), rf(rf_), rh(rh_), lh(lh_), 
Llf(Llf_), Lrf(Lrf_), Lrh(Lrh_), Llh(Llh_),
Glf(Glf_), Grf(Grf_), Grh(Grh_), Glh(Glh_) {
    
}

void states::predict(Eigen::Vector3d a, Eigen::Matrix3d R, Eigen::Vector3d dba, Eigen::Vector3d dbw, double dt) {
    ba += dba;
    bw += dbw;
    p = p + v * dt + R * (a - ba) * dt * dt * 0.5;
    v = v + R * (a - ba) * dt;
    lf.predict(R, Glf, dt);
    rf.predict(R, Grf, dt);
    rh.predict(R, Grh, dt);
    lh.predict(R, Glh, dt);
}

double states::validate_leg(Eigen::Vector3d p, Eigen::Matrix3d R, bool contact, bool slip, leg_states &leg, ground g) {
    // position and contact point differ (only valid when contact and no slip)
    // gnd k and gnd k+1 (when update ground)
    // gnd and contact point distance (valid when swing, should be large as possible)
    // position and contact point differ with larger covariance when contact but slip
    double weight;
    if (contact) { 
        Eigen::DiagonalMatrix<double, 3, 3> p_covariance = slip?
        Eigen::DiagonalMatrix<double, 3, 3>(2.5e-5, 2.5e-5, 2.5e-5): Eigen::DiagonalMatrix<double, 3, 3>(1e-6, 1e-6, 1e-6) ;
        std::vector<double> weights;
        for (int rim = 1; rim <= 5; rim ++ ) {
            Eigen::Vector3d center_point = leg.lookup_point(RIM(rim));
            Eigen::Vector3d contact_center_point_world_frame = p + R * center_point;
            Eigen::Vector3d diff = contact_center_point_world_frame - leg.lookup_predicted_contact_point(RIM(rim));
            weights.push_back(gaussianLikelihood(diff, Eigen::Vector3d(0, 0, 0), p_covariance));
        }
        auto rim_pointer = std::max_element(weights.begin(), weights.end());
        RIM rim_contact = RIM(std::distance(weights.begin(), rim_pointer));
        for (int rim = 1; rim <= 5; rim ++ ) {
            if (rim == rim_contact) continue;
            else leg.assign_contact_point(p, R, RIM(rim));
        }
        weight = *std::max_element(weights.begin(), weights.end());
    }
    else {
        // swing phase
        std::vector<double> weights;
        for (int rim = 1; rim <= 5; rim ++ ) {
            Eigen::Vector3d center_point = leg.lookup_point(RIM(rim));
            double radius = rim == G_POINT? leg.leg.radius() : leg.leg.Radius() + leg.leg.radius();
            double dist_from_gnd = (p + R * center_point - g.position).dot(g.orient);
            double center_from_gnd = (p - g.position).dot(g.orient);
            weights.push_back(relu(dist_from_gnd - radius, center_from_gnd - 0.07, center_from_gnd + 0.061394));
            leg.assign_contact_point(p, R, RIM(rim));
        }
        weight = *std::max_element(weights.begin(), weights.end());
    }

    return weight;
}

double states::validate_ground(Eigen::Vector3d p, Eigen::Matrix3d R, ground g, lidar l, lidar rl, lidar fh) {
    ground new_ground = l.body_frame_ground(rl, fh);
    new_ground = new_ground.coordinate(R, p);
    Eigen::DiagonalMatrix<double, 2, 2> p_covariance(1e-2, 2.5e-5);
    Eigen::Vector2d diff(acos(new_ground.orient.dot(g.orient)) ,new_ground.position(2) - g.position(2));
    return gaussianLikelihood(diff, Eigen::Vector2d(0, 0), p_covariance);
}

}
