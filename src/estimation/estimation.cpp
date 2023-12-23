#include "Estimation.hpp"
#include <iostream>
#include "csv_reader.hpp"
#include "Estimator.hpp"

int main() {
    std::string filename = "data/data07";
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;
    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.012);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.012);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.012);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.012);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 30);
    Estimator est(20, 0.005);
    const size_t N = 20;
    Eigen::Vector3d true_value_estimate;
    for (int i = 1; i < n - 1 - N; i++) {
        Eigen::Matrix4d encoders;
        encoders.row(0) = Eigen::Vector4d(df.iloc("lf.theta", i), df.iloc("lf.beta", i), df.iloc("lf.theta_d", i), df.iloc("lf.beta_d", i));
        encoders.row(1) = Eigen::Vector4d(df.iloc("rf.theta", i), df.iloc("rf.beta", i), df.iloc("rf.theta_d", i), df.iloc("rf.beta_d", i));
        encoders.row(2) = Eigen::Vector4d(df.iloc("rh.theta", i), df.iloc("rh.beta", i), df.iloc("rh.theta_d", i), df.iloc("rh.beta_d", i));
        encoders.row(3) = Eigen::Vector4d(df.iloc("lh.theta", i), df.iloc("lh.beta", i), df.iloc("lh.theta_d", i), df.iloc("lh.beta_d", i));
        est.Input(Eigen::Vector3d(df.iloc("a.x", i+20), df.iloc("a.y", i+20), df.iloc("a.z", i+20)),
            Eigen::Vector3d(df.iloc("w.x", i+20), df.iloc("w.y", i+20), df.iloc("w.z", i+20)), 
            Eigen::Vector4d(df.iloc("q.x", i+20), df.iloc("q.y", i+20), df.iloc("q.z", i+20), df.iloc("q.w", i+20)),
            encoders, Eigen::Vector4d(0,0,0,0)
            );
        double error = 0;
        est.Estimate(true_value_estimate, lf_leg, rf_leg, rh_leg, lh_leg);
        estimate_state.row(i).segment(24, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        estimate_state.row(i).segment(27, 3) = true_value_estimate;
    }
    std::vector<std::string> cols = {"lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z",
    "lf_.x", "lf_.y", "lf_.z", "rf_.x", "rf_.y", "rf_.z", "rh_.x", "rh_.y", "rh_.z", "lh_.x", "lh_.y", "lh_.z",
    "v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    std::cout << estimate_state.block(0, 0, 10, 1) << "\n";
    return 0;
}