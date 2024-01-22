#ifndef CONTACT_MAP_HPP
#define CONTACT_MAP_HPP

#include "kinematic/Leg.hpp"
#include <vector>

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
            // std::cout << bound_beta << "\t" << beta << "\t" << beta2 << "\t" << rim << "\t" << rim2 << "\n\n";
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
    private:
        inline double b1(double theta) {return -2.61019580e-09 * pow(theta, 5) + 1.24181267e-06 * pow(theta, 4) 
        - 2.24183011e-04 * pow(theta, 3) + 1.78431692e-02 * theta * theta - 1.33151836e-01 * theta - 1.78362899e+00 ;}
        inline double b2(double theta) {return -1.22581785e-09 * pow(theta, 5) + 5.02932993e-07 * pow(theta, 4) 
        -7.37114643e-05 * pow(theta, 3) + 6.47617996e-03 * theta * theta -3.31750539e-01 * theta + 5.40846840e+01 ;}
        inline double b3(double theta) {return -4.87190741e-07 * pow(theta, 5) + 3.21347467e-04 * pow(theta, 4) 
        -8.40604260e-02 * pow(theta, 3) + 1.09041600e+01 * theta * theta -7.02946587e+02 * theta + 1.82438639e+04 ;}
};

#endif