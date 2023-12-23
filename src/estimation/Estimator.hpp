#include <Eigen/Dense>
#include "ContactMap.hpp"

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
        }
        void Input(Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Vector4d q, Eigen::Matrix4d eds, Eigen::Vector4d alpha) {
            if (N > 0) {
                for (int i = 0; i < N - 1; ++i) {
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
            A.row(N - 1) = a;
            W.row(N - 1) = w;
            Q.row(N - 1) = q;
            LF.row(N - 1) = eds.row(0);
            RF.row(N - 1) = eds.row(1);
            RH.row(N - 1) = eds.row(2);
            LH.row(N - 1) = eds.row(3);
            APH.row(N - 1) = alpha;
        }
        void Estimate(Eigen::Vector3d &velocity, Leg &lf, Leg &rf, Leg &rh, Leg &lh) {
            Eigen::Vector3d lf_v, rf_v, rh_v, lh_v;
            this->estimate(lf_v, lf, this->LF, this->APH.row(0)(0));
            this->estimate(rf_v, rf, this->RF, this->APH.row(0)(1));
            this->estimate(rh_v, rh, this->RH, this->APH.row(0)(2));
            this->estimate(lh_v, lh, this->LH, this->APH.row(0)(3));
            velocity = lf_v;
        }
        
    private:
        size_t N = 1;
        double dt = 0.001;
        Eigen::Matrix<double, Eigen::Dynamic, 3> A;
        Eigen::Matrix<double, Eigen::Dynamic, 3> W;
        Eigen::Matrix<double, Eigen::Dynamic, 4> Q;
        Eigen::Matrix<double, Eigen::Dynamic, 4> LF;
        Eigen::Matrix<double, Eigen::Dynamic, 4> RF;
        Eigen::Matrix<double, Eigen::Dynamic, 4> RH;
        Eigen::Matrix<double, Eigen::Dynamic, 4> LH;
        Eigen::Matrix<double, Eigen::Dynamic, 4> APH;
        void estimate(Eigen::Vector3d &velocity, Leg &leg, Eigen::Matrix<double, Eigen::Dynamic, 4> encoder_data, double alpha_0) {
            Eigen::Vector3d rolling(0, 0, 0);
            Eigen::Vector3d accelerating(0, 0, 0);
            ContactMap cm;
            double contact_beta = encoder_data.row(0)(1) + alpha_0;
            for (int i = 0; i < N; i++) {
                Eigen::Vector4d encoder_k = encoder_data.row(i);
                Eigen::Vector4d q = Q.row(i);
                Eigen::Quaterniond quat = Eigen::Quaterniond(q);
                Eigen::Matrix3d rot = quat.toRotationMatrix();
                contact_beta += (encoder_k(3) + W.row(i)(1)) * dt;
                leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
                accelerating += dt * dt * 0.5 * (2*i + 1) * (rot * (A.row(i)).transpose()).transpose();
                rolling += (dt * rot * leg.RollVelocity(W.row(i), cm.lookup(encoder_k(0), contact_beta), 0)).transpose();
            }
            Eigen::Vector4d encoder_k = encoder_data.row(N-1);
            leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
            leg.PointContact(cm.lookup(encoder_k(0), encoder_k(1)), 0);
            Eigen::Vector4d q = Q.row(N-1);
            Eigen::Quaterniond quat = Eigen::Quaterniond(q);
            Eigen::Matrix3d rot = quat.toRotationMatrix();
            Eigen::Vector3d pose_k = rot * leg.contact_point;
            
            encoder_k = encoder_data.row(0);
            leg.Calculate(encoder_k(0), encoder_k(2), 0, encoder_k(1), encoder_k(3), 0);
            leg.PointContact(cm.lookup(encoder_k(0), contact_beta), 0);
            q = Q.row(0);
            quat = Eigen::Quaterniond(q);
            rot = quat.toRotationMatrix();
            Eigen::Vector3d pose_kn_last = rot * leg.contact_point;
            velocity = (pose_kn_last - pose_k - rolling - accelerating) / ((double) N * dt);
        }
};