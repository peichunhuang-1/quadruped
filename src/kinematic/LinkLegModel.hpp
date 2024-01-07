#ifndef LINKLEGMODEL_HPP
#define LINKLEGMODEL_HPP
#include <cmath>
#include <complex>
#include <iostream>
#include <mutex>
enum RIM
{
    UPPER_RIM_L = 1, // O1_
    LOWER_RIM_L = 2, // O2_
    G_POINT = 3, // G
    LOWER_RIM_R = 4, // O2
    UPPER_RIM_R = 5, // O1
    NO_CONTACT = 0
};

inline double to_degrees(double radians) {
    return radians * (180.0 / M_PI);
}
inline double to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}
class LinkLegModel
{
    public:
    LinkLegModel(double r = 0.0125, double R = 0.1) {
        this->R = R;
        this->r = r;
        this->l1 = 0.8 * this->R;
        this->l2 = 0.9 * this->R;
        this->l3 = 1.3 * this->R;
        this->l4 = 0.4 * this->R;
        this->l5 = 0.88296634 * this->R;
        this->l6 = 0.2 * this->R;
        this->l7 = 2.0 * this->R * sin(to_radians(101.0 / 2.0));
        this->l8 = 2.0 * this->R * sin(to_radians(113.0 / 2.0));
        this->l9 = 2.0 * this->R * sin(to_radians(17.0 / 2.0));
        this->l10 = 2.0 * this->R * sin(to_radians(50.0 / 2.0));
    }
    void calculate(double theta, double theta_d, double theta_dd);
    double inverse(double r, RIM rim);

    std::complex<double> A;
    std::complex<double> B;
    std::complex<double> C;
    std::complex<double> D;
    std::complex<double> F;
    std::complex<double> H;
    std::complex<double> O1;
    std::complex<double> O2;
    std::complex<double> A_;
    std::complex<double> B_;
    std::complex<double> C_;
    std::complex<double> D_;
    std::complex<double> F_;
    std::complex<double> H_;
    std::complex<double> O1_;
    std::complex<double> O2_;
    std::complex<double> E;
    std::complex<double> G;

    std::complex<double> A_d;
    std::complex<double> B_d;
    std::complex<double> C_d;
    std::complex<double> D_d;
    std::complex<double> F_d;
    std::complex<double> O1_d;
    std::complex<double> O2_d;
    std::complex<double> A_d_;
    std::complex<double> B_d_;
    std::complex<double> C_d_;
    std::complex<double> D_d_;
    std::complex<double> F_d_;
    std::complex<double> O1_d_;
    std::complex<double> O2_d_;
    std::complex<double> E_d;
    std::complex<double> G_d;

    std::complex<double> A_dd;
    std::complex<double> B_dd;
    std::complex<double> C_dd;
    std::complex<double> D_dd;
    std::complex<double> F_dd;
    std::complex<double> O1_dd;
    std::complex<double> O2_dd;
    std::complex<double> A_dd_;
    std::complex<double> B_dd_;
    std::complex<double> C_dd_;
    std::complex<double> D_dd_;
    std::complex<double> F_dd_;
    std::complex<double> O1_dd_;
    std::complex<double> O2_dd_;
    std::complex<double> E_dd;
    std::complex<double> G_dd;

    double O1_w = 0;
    double O2_w = 0;
    double O1_w_ = 0;
    double O2_w_ = 0;
    double O1_w_d = 0;
    double O2_w_d = 0;
    double O1_w_d_ = 0;
    double O2_w_d_ = 0;

    double R;
    double r;
    double l1;
    double l2;
    double l3;
    double l4;
    double l5;
    double l6;
    double l7;
    double l8;
    double l9;
    double l10;

    const double min_theta = to_radians(17.0);
    const double max_theta = to_radians(140.0);
    double max_r_G = 1.;
    std::once_flag flag;
    const double to1 = to_radians(39.5);
    const double to2 = - to_radians(65.0);
    const double tf = to_radians(6.0);
    const double th = to_radians(121.0);

    double oe = 0;
    double oe_d = 0;
    double oe_dd = 0;

    double db = 0;
    double db_d = 0;
    double db_dd = 0;
    
    double theta = to_radians(17.0);
    double theta_d = 0;
    double theta_dd = 0;

    double phi = 0;
    double phi_d = 0;
    double phi_dd = 0;
    
    double epsilon = 0;
    double epsilon_d = 0;
    double epsilon_dd = 0;
    
    double theta2 = 0;
    double theta2_d = 0;
    double theta2_dd = 0;

    double rho = 0;
    double rho_d = 0;
    double rho_dd = 0;

    double to = 0;

    void D_phi();
    void D_oe();
    void D_db();
    void D_epsilon();
    void D_theta2();
    void D_rho();
    void D_O1();
    void D_G();
    void D_O2();
    void symmetry();
};
#endif