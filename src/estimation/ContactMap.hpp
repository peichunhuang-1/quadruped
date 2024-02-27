#ifndef CONTACT_MAP_HPP
#define CONTACT_MAP_HPP

#include "kinematic/Leg.hpp"
#include <vector>
#include <deque>
#include <tuple>

using trajectory = std::tuple<double, double, double, Eigen::Matrix3d>; // theta, beta, contact_exp, R

class ContactMap {
    public:
        ContactMap() {}
        RIM lookup(double theta, double beta) { // theta [17  160]; beta [0  360)
            RIM r = NO_CONTACT;
            rad_mod2(beta);
            theta = theta * 180.0 / M_PI;
            beta = beta * 180.0 / M_PI;
            if (theta > 108.3) {
                if (b1(theta) > beta) r = G_POINT;
                else if (b2(theta) >= beta) r = LOWER_RIM_R;
                else if (b3(theta) > beta) r = UPPER_RIM_R;
                else if ((360 - b3(theta)) > beta) r = NO_CONTACT;
                else if ((360 - b2(theta)) > beta) r = UPPER_RIM_L;
                else if ((360 - b1(theta)) > beta) r = LOWER_RIM_L;
                else r = G_POINT;
            }
            else {
                if (b1(theta) > beta) r = G_POINT;
                else if (b2(theta) >= beta) r = LOWER_RIM_R;
                else if (180.0 > beta) r = UPPER_RIM_R;
                else if ((360 - b2(theta)) > beta) r = UPPER_RIM_L;
                else if ((360 - b1(theta)) > beta) r = LOWER_RIM_L;
                else r = G_POINT;
            }
            return r;
        }
        std::pair<double, double> Boundary(double theta, RIM r) {
            switch(r) {
                case G_POINT:
                    return std::pair<double, double>(-b1(theta), b1(theta));
                break;
                case LOWER_RIM_R:
                    return std::pair<double, double>(b1(theta), b2(theta));
                break;
                case UPPER_RIM_R:
                    return theta > 108.3? std::pair<double, double>(b2(theta), b3(theta)) : std::pair<double, double>(b2(theta), 180.0);
                break;
                case NO_CONTACT:
                    return theta > 108.3? std::pair<double, double>(b3(theta), 360 - b3(theta)) : std::pair<double, double>(180.0, 180.0);
                break;
                case UPPER_RIM_L:
                    return theta > 108.3? std::pair<double, double>(360 - b3(theta), 360 - b2(theta)) : std::pair<double, double>(180.0, 360 - b2(theta));
                break;
                case LOWER_RIM_L:
                    return std::pair<double, double>(360 - b2(theta), 360 - b1(theta));
                break;
            }
        }

        double boudary_beta(double theta, double beta, double theta2, double beta2) {
            RIM rim = this->lookup(theta, beta);
            RIM rim2 = this->lookup(theta2, beta2);
            if (rim == rim2) return 0;
            std::pair<double, double> boundary = this->Boundary(theta * 180. / M_PI, rim);
            double bl = boundary.first * M_PI / 180.0; rad_mod2(bl);
            double bu = boundary.second * M_PI / 180.0; rad_mod2(bu);
            double diff1 = (bl - beta); rad_mod(diff1);
            double diff2 = (bu - beta); rad_mod(diff2);
            double diff3 = (bl - beta2); rad_mod(diff3);
            double diff4 = (bu - beta2); rad_mod(diff4);
            diff1 = abs(diff1) > abs(diff3)? diff3: diff1;
            diff2 = abs(diff2) > abs(diff4)? diff4: diff2;
            double bound_beta;
            if (abs(diff1) > abs(diff2)) bound_beta = bu;
            else bound_beta = bl;
            return bound_beta;
        }
        void rad_mod(double &rad) {
            if (rad > M_PI) {
                rad -= 2*M_PI;
                rad_mod(rad);
            }
            else if (rad <= -M_PI) {
                rad += 2*M_PI;
                rad_mod(rad);
            }
        }

        void rad_mod2(double &rad) {
            if (rad > 2. * M_PI) {
                rad -= 2*M_PI;
                rad_mod2(rad);
            }
            else if (rad <= 0) {
                rad += 2*M_PI;
                rad_mod2(rad);
            }
        }

        Eigen::Vector3d travel(std::deque<trajectory> path, Leg &leg) {
            int size = path.size() ;
            if (size == 1) return Eigen::Vector3d(0, 0, 0);
            RIM last_rim = this->lookup(std::get<0>(path[0]), std::get<2>(path[0]));
            Eigen::Vector3d distance(0, 0, 0);
            for (int i = 0; i < size - 1; i ++) {
                RIM current_rim = this->lookup(std::get<0>(path[i+1]), std::get<2>(path[i+1]));
                if (current_rim != last_rim) {
                    if (current_rim == G_POINT || last_rim == G_POINT) {
                        double bound_beta = this->boudary_beta(std::get<0>(path[i]), std::get<2>(path[i]), std::get<0>(path[i+1]), std::get<2>(path[i+1]));
                        Eigen::Vector3d travel_current = travel_contineous(last_rim, leg, std::get<2>(path[i]), bound_beta, std::get<1>(path[i]), std::get<3>(path[i]));
                        distance += travel_current;
                        travel_current = travel_contineous(current_rim, leg, bound_beta, std::get<2>(path[i+1]), std::get<1>(path[i]), std::get<3>(path[i]));
                        distance += travel_current;
                    }
                    else if (current_rim == NO_CONTACT || last_rim == NO_CONTACT) {
                        return Eigen::Vector3d(0, 0, 0);
                    }
                    else {
                        Eigen::Vector3d travel_current = travel_contineous(last_rim, leg, std::get<2>(path[i]), std::get<2>(path[i+1]), std::get<1>(path[i]), std::get<3>(path[i]));
                        distance += travel_current;
                        leg.Calculate(std::get<0>(path[i]), 0, 0, std::get<1>(path[i]), 0, 0);
                        double angle_between_body_frame = std::get<2>(path[i]) - std::get<1>(path[i]);
                        if ( angle_between_body_frame >= M_PI)  angle_between_body_frame -= M_PI * 2;
                        if ( angle_between_body_frame <= -M_PI)  angle_between_body_frame += M_PI * 2;
                        leg.PointContact(last_rim, angle_between_body_frame);
                        Eigen::Vector3d point_from = leg.contact_point;
                        leg.PointContact(current_rim, angle_between_body_frame);
                        Eigen::Vector3d point_to = leg.contact_point;
                        distance += std::get<3>(path[i]) * (point_to - point_from);
                    }
                }
                else {
                    Eigen::Vector3d travel_current = travel_contineous(last_rim, leg, std::get<2>(path[i]), std::get<2>(path[i+1]), std::get<1>(path[i]), std::get<3>(path[i]));
                    distance += travel_current;
                }
                last_rim = current_rim;
            }
            return distance;
        }

        Eigen::Vector3d compensate(std::deque<trajectory> path, Leg &leg, std::deque<double> theta_d, double dt) {
            int size = path.size() ;
            if (size == 1) return Eigen::Vector3d(0, 0, 0);
            RIM last_rim = this->lookup(std::get<0>(path[0]), std::get<2>(path[0]));
            Eigen::Vector3d distance(0, 0, 0);
            for (int i = 0; i < size - 1; i ++) {
                RIM current_rim = this->lookup(std::get<0>(path[i+1]), std::get<2>(path[i+1]));
                if (current_rim == NO_CONTACT || last_rim == NO_CONTACT) {return Eigen::Vector3d(0, 0, 0);}
                else {
                    leg.Calculate(std::get<0>(path[i]), theta_d[i], 0, 0, 0, 0);
                    double omega_last = leg.RimRoll(last_rim);
                    double omega_current = leg.RimRoll(current_rim);
                    double radius_current = current_rim == G_POINT? leg.radius(): leg.Radius() + leg.radius();
                    double radius_last = last_rim == G_POINT? leg.radius(): leg.Radius() + leg.radius();
                    double rolling_distance = (radius_last * omega_last + radius_current * omega_current) * dt * 0.5;
                    double angle_between_body_frame = std::get<2>(path[i]) - std::get<1>(path[i]);
                    if ( angle_between_body_frame >= M_PI)  angle_between_body_frame -= M_PI * 2;
                    if ( angle_between_body_frame <= -M_PI)  angle_between_body_frame += M_PI * 2;
                    Eigen::Vector3d body_frame_travel(cos(angle_between_body_frame) * rolling_distance, 0, sin(-angle_between_body_frame) * rolling_distance);
                    distance += std::get<3>(path[i]) * body_frame_travel;
                }
                last_rim = current_rim;
            }
            return distance;
        }
    private:
        inline double b1(double theta) {return -2.61019580e-09 * pow(theta, 5) + 1.24181267e-06 * pow(theta, 4) 
        - 2.24183011e-04 * pow(theta, 3) + 1.78431692e-02 * theta * theta - 1.33151836e-01 * theta - 1.78362899e+00 ;}
        inline double b2(double theta) {return -1.22581785e-09 * pow(theta, 5) + 5.02932993e-07 * pow(theta, 4) 
        -7.37114643e-05 * pow(theta, 3) + 6.47617996e-03 * theta * theta -3.31750539e-01 * theta + 5.40846840e+01 ;}
        inline double b3(double theta) {return -4.87190741e-07 * pow(theta, 5) + 3.21347467e-04 * pow(theta, 4) 
        -8.40604260e-02 * pow(theta, 3) + 1.09041600e+01 * theta * theta -7.02946587e+02 * theta + 1.82438639e+04 ;}
        Eigen::Vector3d travel_contineous(RIM current_rim, Leg &leg, double contact_beta_from, double contact_beta_to, double beta_from, Eigen::Matrix3d rot_from) {
            double radius = current_rim == G_POINT? leg.radius(): current_rim == NO_CONTACT? 0: leg.radius() + leg.Radius();
            double dbeta = contact_beta_to - contact_beta_from;
            if (dbeta >= M_PI) dbeta -= M_PI * 2;
            if (dbeta <= -M_PI) dbeta += M_PI * 2;
            double angle_between_body_frame = contact_beta_from - beta_from;
            if ( angle_between_body_frame >= M_PI)  angle_between_body_frame -= M_PI * 2;
            if ( angle_between_body_frame <= -M_PI)  angle_between_body_frame += M_PI * 2;
            double d = radius * dbeta;
            Eigen::Vector3d body_frame_travel(cos(angle_between_body_frame) * d, 0, sin(-angle_between_body_frame) * d) ;
            return rot_from * body_frame_travel;
        }
};

#endif