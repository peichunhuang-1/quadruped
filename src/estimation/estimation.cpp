#include <iostream>
#include "csv_reader.hpp"
#include "Estimation.hpp"
#include "Estimator.hpp"

int main(int argc, char* argv[]) {
    std::string filenumber = std::string(argv[1]);
    std::string filename = "data/data" + filenumber;
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;
    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.012);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.012);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.012);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.012);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 30);
    // const size_t N = 15;
    // Estimator est(N, 0.005);
    Eigen::Vector3d true_value_estimate;
    for (int i = 1; i < n - 1; i++) {
        Eigen::Matrix4d encoders;
        encoders.row(0) = Eigen::Vector4d(df.iloc("lf.theta", i), df.iloc("lf.beta", i), df.iloc("lf.theta_d", i), df.iloc("lf.beta_d", i));
        encoders.row(1) = Eigen::Vector4d(df.iloc("rf.theta", i), df.iloc("rf.beta", i), df.iloc("rf.theta_d", i), df.iloc("rf.beta_d", i));
        encoders.row(2) = Eigen::Vector4d(df.iloc("rh.theta", i), df.iloc("rh.beta", i), df.iloc("rh.theta_d", i), df.iloc("rh.beta_d", i));
        encoders.row(3) = Eigen::Vector4d(df.iloc("lh.theta", i), df.iloc("lh.beta", i), df.iloc("lh.theta_d", i), df.iloc("lh.beta_d", i));
        double alpha_l = atan2((df.iloc("lf.dist", i) - df.iloc("lh.dist", i)) , 0.4);
        double alpha_r = atan2((df.iloc("rf.dist", i) - df.iloc("rh.dist", i)) , 0.4);
        Eigen::Vector3d w = Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i));
        Eigen::Vector3d a = Eigen::Vector3d(df.iloc("a.x", i), df.iloc("a.y", i), df.iloc("a.z", i));
        Eigen::Vector4d q = Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i));
        // est.Input(Eigen::Vector3d(df.iloc("a.x", i+N), df.iloc("a.y", i+N), df.iloc("a.z", i+N)),
        //     Eigen::Vector3d(df.iloc("w.x", i+N), df.iloc("w.y", i+N), df.iloc("w.z", i+N)), 
        //     Eigen::Vector4d(df.iloc("q.x", i+N), df.iloc("q.y", i+N), df.iloc("q.z", i+N), df.iloc("q.w", i+N)),
        //     encoders, Eigen::Vector4d(alpha_l, alpha_r, alpha_r, alpha_l)
        // );
        double error_lf, error_rf, error_rh, error_lh = 0;
        // est.Estimate(true_value_estimate, lf_leg, rf_leg, rh_leg, lh_leg, error_lf, error_rf, error_rh, error_lh);
        double contact_lf, contact_rf, contact_rh, contact_lh;
        Eigen::Vector3d lf_v, rf_v, rh_v, lh_v;
        lf_v = velocity2(encoders.row(0), w, q, lf_leg, alpha_l);
        rf_v = velocity2(encoders.row(0), w, q, rf_leg, alpha_r);
        rh_v = velocity2(encoders.row(0), w, q, rh_leg, alpha_r);
        lh_v = velocity2(encoders.row(0), w, q, lh_leg, alpha_l);

        contact_lf = contact_prob(lf_leg, df.iloc("lf.theta", i), df.iloc("lf.beta", i), df.iloc("lf.dist", i), alpha_l);
        contact_rf = contact_prob(rf_leg, df.iloc("rf.theta", i), df.iloc("rf.beta", i), df.iloc("rf.dist", i), alpha_r);
        contact_rh = contact_prob(rh_leg, df.iloc("rh.theta", i), df.iloc("rh.beta", i), df.iloc("rh.dist", i), alpha_r);
        contact_lh = contact_prob(lh_leg, df.iloc("lh.theta", i), df.iloc("lh.beta", i), df.iloc("lh.dist", i), alpha_l);

        estimate_state.row(i).segment(0, 4) = Eigen::Vector4d(error_lf, error_rf, error_rh, error_lh);
        estimate_state.row(i).segment(4, 4) = Eigen::Vector4d(df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i));
        estimate_state.row(i).segment(8, 4) = Eigen::Vector4d(contact_lf , contact_rf , contact_rh , contact_lh );
        estimate_state.row(i).segment(12, 3) = lf_v;
        estimate_state.row(i).segment(15, 3) = rf_v;
        estimate_state.row(i).segment(18, 3) = rh_v;
        estimate_state.row(i).segment(21, 3) = lh_v;

        estimate_state.row(i).segment(24, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        // estimate_state.row(i).segment(27, 3) = true_value_estimate;
    }
    std::vector<std::string> cols = {"lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z",
    "lf_.x", "lf_.y", "lf_.z", "rf_.x", "rf_.y", "rf_.z", "rh_.x", "rh_.y", "rh_.z", "lh_.x", "lh_.y", "lh_.z",
    "v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    std::cout << estimate_state.block(0, 0, 10, 1) << "\n";
    return 0;
}