#include <Eigen/Dense>
#include "ContactMap.hpp"

double normal_pdf(double x, double m, double s) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;
    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

double sigmoid(double x) {
    return (std::erf(x) + 1) * 0.5;
}

double contact_prob(Leg &leg, double theta, double beta, double dst, double alpha = 0) {
    ContactMap cm;
    leg.Calculate(theta, 0, 0, beta, 0, 0);
    leg.PointContact(cm.lookup(theta, beta), alpha);
    std::cout << dst << "\t" << -leg.contact_point(2) << "\n";
    double dst_to_ground = -leg.contact_point(2);
    double sigma = (0.01 + abs(leg.contact_point(0)) * 0.1);
    double p = sigmoid((dst_to_ground - dst) / sqrt(2) / sigma / sigma) > 0.5 ? 0 : 1; 
    return sigmoid((dst_to_ground - dst) / sqrt(2) / sigma / sigma);
}

class Estimator {
    public:
        Estimator(size_t n, double dt_ = 0.001) : N(n), dt(dt_) {
            A.resize(N, 3);
            W.resize(N, 3);
            Q.resize(N, 4);
            LF.resize(N, 4);
            RF.resize(N, 4);
            RH.resize(N, 4);
            LH.resize(N, 4);
            APH.resize(N, 4);
            dV.resize(N, 3);
            Vlf.resize(N, 3);
            Vrf.resize(N, 3);
            Vrh.resize(N, 3);
            Vlh.resize(N, 3);
        }
        void Input(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Vector4d q, Eigen::Matrix4d eds, Eigen::Vector4d alpha) {
            if (N > 0) {
                for (int i = 0; i < N - 1; ++i) {
                    dV.row(i) = dV.row(i + 1);
                    Vlf.row(i) = Vlf.row(i + 1);
                    Vrf.row(i) = Vrf.row(i + 1);
                    Vrh.row(i) = Vrh.row(i + 1);
                    Vlh.row(i) = Vlh.row(i + 1);
                    A.row(i) = A.row(i + 1);
                    W.row(i) = W.row(i + 1);
                    Q.row(i) = Q.row(i + 1);
                    LF.row(i) = LF.row(i + 1);
                    RF.row(i) = RF.row(i + 1);
                    RH.row(i) = RH.row(i + 1);
                    LH.row(i) = LH.row(i + 1);
                    APH.row(i) = APH.row(i + 1);
                }
            }
            Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            dV.row(N - 1) = (rot * (a - Eigen::Vector3d(1e-2, 1e-2, 1e-2)) * dt).transpose();
            A.row(N - 1) = a - Eigen::Vector3d(1e-2, 1e-2, 1e-2);
            W.row(N - 1) = w - Eigen::Vector3d(1e-3, 1e-3, 1e-3);
            Q.row(N - 1) = q;
            LF.row(N - 1) = eds.row(0);
            RF.row(N - 1) = eds.row(1);
            RH.row(N - 1) = eds.row(2);
            LH.row(N - 1) = eds.row(3);
            APH.row(N - 1) = alpha;
        }
        void Estimate(Eigen::Vector3d &velocity, Leg &lf, Leg &rf, Leg &rh, Leg &lh, double &err_lf, double &err_rf, double &err_rh, double &err_lh) {
            Eigen::Vector3d lf_v, rf_v, rh_v, lh_v;
            this->estimate(lf_v, lf, this->LF, this->APH.row(0)(0));
            this->estimate(rf_v, rf, this->RF, this->APH.row(0)(1));
            this->estimate(rh_v, rh, this->RH, this->APH.row(0)(2));
            this->estimate(lh_v, lh, this->LH, this->APH.row(0)(3));
            Vlf.row(N - 1) = lf_v; // k - n
            Vrf.row(N - 1) = rf_v;
            Vrh.row(N - 1) = rh_v;
            Vlh.row(N - 1) = lh_v;
            err_lf = error(Vlf);
            err_rf = error(Vrf);
            err_rh = error(Vrh);
            err_lh = error(Vlh);
            double pdf_lf = normal_pdf(err_lf, 0, 0.08);
            double pdf_rf = normal_pdf(err_rf, 0, 0.08);
            double pdf_rh = normal_pdf(err_rh, 0, 0.08);
            double pdf_lh = normal_pdf(err_lh, 0, 0.08);
            double total = pdf_lf + pdf_rf + pdf_rh + pdf_lh;
            if (total > 1e-4) velocity = (pdf_lf * lf_v + pdf_rf * rf_v + pdf_rh * rh_v + pdf_lh * lh_v) / total;
        }
        void estimate(Eigen::Vector3d &velocity, Leg &leg, Eigen::Matrix<double, Eigen::Dynamic, 4> encoder_data, double alpha_0) {
            // Eigen::Vector3d rolling(0, 0, 0);
            // Eigen::Vector3d accelerating(0, 0, 0);
            // ContactMap cm;
            // double contact_beta = encoder_data.row(0)(1) + alpha_0;
            // for (int i = 0; i < N; i++) {
            //     Eigen::Vector4d encoder_k = encoder_data.row(i);
            //     Eigen::Vector4d q = Q.row(i);
            //     Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            //     Eigen::Matrix3d rot = quat.toRotationMatrix();
            //     contact_beta += (encoder_k(3) + W.row(i)(1)) * dt;
            //     leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
            //     accelerating += dt * dt * 0.5 * (2*i + 1) * (rot * (A.row(i)).transpose()).transpose();
            //     rolling += (dt * rot * leg.RollVelocity(W.row(i), cm.lookup(encoder_k(0), contact_beta), 0)).transpose();
            // }
            // Eigen::Vector4d encoder_k = encoder_data.row(N-1);
            // leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
            // leg.PointContact(cm.lookup(encoder_k(0), encoder_k(1)), 0);
            // Eigen::Vector4d q = Q.row(N-1);
            // Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            // Eigen::Matrix3d rot = quat.toRotationMatrix();
            // Eigen::Vector3d pose_k = rot * leg.contact_point;
            
            // encoder_k = encoder_data.row(0);
            // leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
            // leg.PointContact(cm.lookup(encoder_k(0), contact_beta), 0);
            // q = Q.row(0);
            // quat = Eigen::Quaterniond(q);
            // rot = quat.toRotationMatrix();
            // Eigen::Vector3d pose_kn_last = rot * leg.contact_point;
            // velocity = (pose_kn_last - pose_k - rolling - accelerating) / ((double) N * dt);
            ContactMap cm;
            leg.Calculate(encoder_data.row(N - 1)(0), encoder_data.row(N - 1)(2), 0, encoder_data.row(N - 1)(1), encoder_data.row(N - 1)(3), 0);
            leg.PointVelocity(Eigen::Vector3d(0, 0, 0), W.row(N - 1), cm.lookup(encoder_data.row(N - 1)(0), encoder_data.row(N - 1)(1)), 0);
            Eigen::Vector4d quaternion = Q.row(N-1);
            Eigen::Quaterniond quat = Eigen::Quaterniond(quaternion);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            velocity = - rot * leg.contact_velocity;
        }
    private:
        size_t N = 1;
        double dt = 0.001;
        Eigen::Matrix<double, Eigen::Dynamic, 3> dV;
        Eigen::Matrix<double, Eigen::Dynamic, 3> Vlf;
        Eigen::Matrix<double, Eigen::Dynamic, 3> Vrf;
        Eigen::Matrix<double, Eigen::Dynamic, 3> Vrh;
        Eigen::Matrix<double, Eigen::Dynamic, 3> Vlh;
        Eigen::Matrix<double, Eigen::Dynamic, 3> A;
        Eigen::Matrix<double, Eigen::Dynamic, 3> W;
        Eigen::Matrix<double, Eigen::Dynamic, 4> Q;
        Eigen::Matrix<double, Eigen::Dynamic, 4> LF;
        Eigen::Matrix<double, Eigen::Dynamic, 4> RF;
        Eigen::Matrix<double, Eigen::Dynamic, 4> RH;
        Eigen::Matrix<double, Eigen::Dynamic, 4> LH;
        Eigen::Matrix<double, Eigen::Dynamic, 4> APH;
        double error(Eigen::Matrix<double, Eigen::Dynamic, 3> v) {
            Eigen::Vector3d sum_of_a = {0, 0, 0};
            // for (int i = 0; i < N - 2; i++) {
            //     sum_of_a += dV.row(i);
            // }
            // Eigen::Vector3d dv = v.row(N - 1) - v.row(0) - sum_of_a.transpose();
            for (int i = 0; i < N - 2; i++) {
                sum_of_a += v.row(i + 1) - v.row(i) - dV.row(i);
            }

            // return dv.norm();
            return sum_of_a.norm();
        }
};