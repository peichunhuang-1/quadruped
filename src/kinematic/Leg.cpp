#include "Leg.hpp"
void Leg::Calculate(double theta, double theta_d, double theta_dd, double beta, double beta_d, double beta_dd)
{
    this->beta = beta;
    this->beta_d = beta_d;
    this->beta_dd = beta_dd;
    std::complex<double> rot_ang =  std::polar(1., beta);
    std::complex<double> rot_vel =  beta_d * std::polar(1., beta + M_PI_2);
    this->calculate(theta, theta_d, theta_dd);
    this->O1 = rot_ang * this->O1;
    this->O1_ = rot_ang * this->O1_;
    this->O2 = rot_ang * this->O2;
    this->O2_ = rot_ang * this->O2_;
    this->O1_d = rot_ang * this->O1_d + rot_vel * this->O1;
    this->O1_d_ = rot_ang * this->O1_d_ + rot_vel * this->O1_;
    this->O2_d = rot_ang * this->O2_d + rot_vel * this->O2;
    this->O2_d_ = rot_ang * this->O2_d_ + rot_vel * this->O2_;
    this->G = rot_ang * this->G;
    this->F = rot_ang * this->F;
    this->F_ = rot_ang * this->F_;
    this->H = rot_ang * this->H;
    this->H_ = rot_ang * this->H_;
    this->G_d = rot_ang * this->G_d + rot_vel * this->G;
    this->O1_w += beta_d;
    this->O1_w_ += beta_d;
    this->O2_w += beta_d;
    this->O2_w_ += beta_d;
}

Eigen::Vector3d Leg::RimCentorVelocity(Eigen::Vector3d v, Eigen::Vector3d w, RIM rim) {
    switch (rim) {
        case G_POINT:
        {
            return v + w.cross(this->offset + Eigen::Vector3d(this->G.imag(), 0, this->G.real())) + Eigen::Vector3d(this->G_d.imag(), 0, this->G_d.real());
        break;
        }
        case UPPER_RIM_R:
        {
            return  v + w.cross(this->offset + Eigen::Vector3d(this->O1.imag(), 0, this->O1.real())) + Eigen::Vector3d(this->O1_d.imag(), 0, this->O1_d.real());
        break;
        }
        case LOWER_RIM_R:
        {
            return v + w.cross(this->offset + Eigen::Vector3d(this->O2.imag(), 0, this->O2.real())) + Eigen::Vector3d(this->O2_d.imag(), 0, this->O2_d.real());
        break;
        }
        case LOWER_RIM_L:
        {
            return v + w.cross(this->offset + Eigen::Vector3d(this->O2_.imag(), 0, this->O2_.real())) + Eigen::Vector3d(this->O2_d_.imag(), 0, this->O2_d_.real());
        break;
        }
        case UPPER_RIM_L:
        {
            return v + w.cross(this->offset + Eigen::Vector3d(this->O1_.imag(), 0, this->O1_.real())) + Eigen::Vector3d(this->O1_d_.imag(), 0, this->O1_d_.real());
        break;
        }
        default:
        {
            return v + w.cross(this->offset);
        }
        break;
    }
}

Eigen::Vector3d Leg::RimCentorPosition(RIM rim) {
    switch (rim) {
        case G_POINT:
        {
            return this->offset + Eigen::Vector3d(this->G.imag(), 0, this->G.real());
        break;
        }
        case UPPER_RIM_R:
        {
            return  this->offset + Eigen::Vector3d(this->O1.imag(), 0, this->O1.real());
        break;
        }
        case LOWER_RIM_R:
        {
            return this->offset + Eigen::Vector3d(this->O2.imag(), 0, this->O2.real());
        break;
        }
        case LOWER_RIM_L:
        {
            return this->offset + Eigen::Vector3d(this->O2_.imag(), 0, this->O2_.real());
        break;
        }
        case UPPER_RIM_L:
        {
            return this->offset + Eigen::Vector3d(this->O1_.imag(), 0, this->O1_.real());
        break;
        }
        default:
        {
            return this->offset;
        }
        break;
    }
}

void Leg::PointContact(RIM rim, double alpha) {
    double rim_radius = rim == G_POINT? this->r : this->r + this->R;
    std::complex<double> rim_p = std::polar(rim_radius, M_PI + alpha);
    switch (rim) {
        case G_POINT:
        {
            this->contact_point = this->offset + Eigen::Vector3d(this->G.imag() + rim_p.imag(), 0, this->G.real() + rim_p.real());
        break;
        }
        case UPPER_RIM_R:
        {
            this->contact_point = this->offset + Eigen::Vector3d(this->O1.imag() + rim_p.imag(), 0, this->O1.real() + rim_p.real());
        break;
        }
        case LOWER_RIM_R:
        {
            this->contact_point = this->offset + Eigen::Vector3d(this->O2.imag() + rim_p.imag(), 0, this->O2.real() + rim_p.real());
        break;
        }
        case LOWER_RIM_L:
        {
            this->contact_point = this->offset + Eigen::Vector3d(this->O2_.imag() + rim_p.imag(), 0, this->O2_.real() + rim_p.real());
        break;
        }
        case UPPER_RIM_L:
        {
            this->contact_point = this->offset + Eigen::Vector3d(this->O1_.imag() + rim_p.imag(), 0, this->O1_.real() + rim_p.real());
        break;
        }
        default:
        {
            this->contact_point = this->offset;
        }
        break;
    }
}

void Leg::PointVelocity(Eigen::Vector3d v, Eigen::Vector3d w, RIM rim, double alpha, bool inbody_coord) {
    double rim_radius = rim == G_POINT? this->r : this->r + this->R;
    if (this->offset(1) < 0) link_w = O2_w_; // right side leg, left side lower rim
    else link_w = O2_w;
    rim_p = std::polar(rim_radius, M_PI + alpha);
    switch (rim) {
        case G_POINT:
        {
            this->contact_velocity = v + w.cross(this->contact_point) + Eigen::Vector3d(this->G_d.imag(), 0, this->G_d.real()) + Eigen::Vector3d(0, link_w, 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case UPPER_RIM_R:
        {
            link_w = O1_w;
            this->contact_velocity = v + w.cross(this->contact_point) + Eigen::Vector3d(this->O1_d.imag(), 0, this->O1_d.real()) + Eigen::Vector3d(0, link_w, 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case LOWER_RIM_R:
        {
            link_w = O2_w;
            this->contact_velocity = v + w.cross(this->contact_point) + Eigen::Vector3d(this->O2_d.imag(), 0, this->O2_d.real()) + Eigen::Vector3d(0, link_w, 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case LOWER_RIM_L:
        {
            link_w = O2_w_;
            this->contact_velocity = v + w.cross(this->contact_point) + Eigen::Vector3d(this->O2_d_.imag(), 0, this->O2_d_.real()) + Eigen::Vector3d(0, link_w, 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case UPPER_RIM_L:
        {
            link_w = O1_w_;
            this->contact_velocity = v + w.cross(this->contact_point) + Eigen::Vector3d(this->O1_d_.imag(), 0, this->O1_d_.real()) + Eigen::Vector3d(0, link_w, 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        default:
        {
            link_w = 0;
            this->contact_velocity = v + w.cross(this->contact_point);
        }
        break;
    }
    if (!inbody_coord) {
        std::complex<double> normal_vec = std::polar(1., alpha);
        std::complex<double> tangent_vec = std::polar(1., alpha + M_PI_2);
        this->contact_velocity = Eigen::Vector3d(this->contact_velocity.dot(Eigen::Vector3d(tangent_vec.imag(), 0, tangent_vec.real())), this->contact_velocity(1), this->contact_velocity.dot(Eigen::Vector3d(normal_vec.imag(), 0, normal_vec.real())));
    }
}

double Leg::RimRoll(RIM rim) {
    if (this->offset(1) < 0) {
        link_w = O2_w_; // right side leg, left side lower rim
        link_w_d = O2_w_d_;
    }
    else {
        link_w = O2_w;
        link_w_d = O2_w_d_;
    }
    switch (rim) {
        case G_POINT:
        {
        break;
        }
        case UPPER_RIM_R:
        {
            link_w = O1_w;
            link_w_d = O1_w_d;
        break;
        }
        case LOWER_RIM_R:
        {
            link_w = O2_w;
            link_w_d = O2_w_d;
        break;
        }
        case LOWER_RIM_L:
        {
            link_w = O2_w_;
            link_w_d = O2_w_d_;
        break;
        }
        case UPPER_RIM_L:
        {
            link_w = O1_w_;
            link_w_d = O1_w_d_;
        break;
        }
        default:
        {
            link_w = 0;
            link_w_d = 0;
        }
        break;
    }
    return link_w;
}

Eigen::Vector3d Leg::RollVelocity(Eigen::Vector3d w, RIM rim, double alpha) {
    double rim_radius = rim == G_POINT? this->r : this->r + this->R;
    if (this->offset(1) < 0) {
        link_w = O2_w_; // right side leg, left side lower rim
        link_w_d = O2_w_d_;
    }
    else {
        link_w = O2_w;
        link_w_d = O2_w_d_;
    }
    rim_p = std::polar(rim_radius, M_PI + alpha);
    switch (rim) {
        case G_POINT:
        {
            return Eigen::Vector3d(0, link_w + w(1), 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case UPPER_RIM_R:
        {
            link_w = O1_w;
            link_w_d = O1_w_d;
            return Eigen::Vector3d(0, link_w + w(1), 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case LOWER_RIM_R:
        {
            link_w = O2_w;
            link_w_d = O2_w_d;
            return Eigen::Vector3d(0, link_w + w(1), 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case LOWER_RIM_L:
        {
            link_w = O2_w_;
            link_w_d = O2_w_d_;
            return Eigen::Vector3d(0, link_w + w(1), 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        case UPPER_RIM_L:
        {
            link_w = O1_w_;
            link_w_d = O1_w_d_;
            return Eigen::Vector3d(0, link_w + w(1), 0).cross(Eigen::Vector3d(rim_p.imag(), 0, rim_p.real()));
        break;
        }
        default:
        {
            link_w = 0;
            link_w_d = 0;
            return Eigen::Vector3d(0, 0, 0);
        }
        break;
    }
}

std::pair<double, double> Leg::Inverse(Eigen::Vector3d p, RIM rim=G_POINT) {
    p = p - this->offset;
    std::pair<double, double> motor_angles;
    double r = sqrt(p(0) * p(0) + p(2) * p(2));
    motor_angles.first = this->inverse(r, rim); // theta: 開合角
    motor_angles.second = atan2(-p(0), -p(2)); // beta: 腳的轉角
    return motor_angles;
}