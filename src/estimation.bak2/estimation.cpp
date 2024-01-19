#include <iostream>
#include "csv_reader.hpp"
#include "ObservationModel.hpp"

int main(int argc, char* argv[]) {
    std::string filenumber = std::string(argv[1]);
    std::string filename = "data/data" + filenumber;
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;
    LegVelocityEstimation lf_leg(Eigen::Vector3d(0.2, 0.15, 0), Eigen::Vector3d(0.2 , 0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rf_leg(Eigen::Vector3d(0.2, -0.15, 0), Eigen::Vector3d(0.2 ,-0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), Eigen::Vector3d(-0.2 ,-0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), Eigen::Vector3d(-0.2 ,0.08, 0), 0.1, 0.01, 0.005);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 30);
    Eigen::Vector3d true_value_estimate;
    int counter = 0;
    // VelocityOptimizer fopt_p;
    // nlopt::opt fopt = Optimizer(&fopt_p);
    // std::vector<double> u {0, 0, 0};
    // double minf = 0;

    for (int i = 1; i < n - 1; i++) {
        bool update = counter % 20 == 0? true: false;
        ENCODER_DATA elf = {
            df.iloc("lf.beta", i),
            df.iloc("lf.theta", i),
            df.iloc("lf.beta_d", i),
            df.iloc("lf.theta_d", i),
        };
        ENCODER_DATA erf = {
            df.iloc("rf.beta", i),
            df.iloc("rf.theta", i),
            df.iloc("rf.beta_d", i),
            df.iloc("rf.theta_d", i),
        };
        ENCODER_DATA erh = {
            df.iloc("rh.beta", i),
            df.iloc("rh.theta", i),
            df.iloc("rh.beta_d", i),
            df.iloc("rh.theta_d", i),
        };
        ENCODER_DATA elh = {
            df.iloc("lh.beta", i),
            df.iloc("lh.theta", i),
            df.iloc("lh.beta_d", i),
            df.iloc("lh.theta_d", i),
        };

        double alpha_l = atan2((df.iloc("lf.dist", i) - df.iloc("lh.dist", i)) , 0.4);
        double alpha_r = atan2((df.iloc("rf.dist", i) - df.iloc("rh.dist", i)) , 0.4);
        DST_DATA dlf = {
            df.iloc("lf.dist", i),
            alpha_l
        };
        DST_DATA drf = {
            df.iloc("rf.dist", i),
            alpha_r
        };
        DST_DATA drh = {
            df.iloc("rh.dist", i),
            alpha_r
        };
        DST_DATA dlh = {
            df.iloc("lh.dist", i),
            alpha_l
        };
        IMU_DATA imu = {
            Eigen::Vector3d(df.iloc("a.x", i), df.iloc("a.y", i), df.iloc("a.z", i)),
            Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i)),
            Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i))
        };
        // lf_leg.calculate(imu, elf, dlf, update);
        // rf_leg.calculate(imu, erf, drf, update);
        // rh_leg.calculate(imu, erh, drh, update);
        // lh_leg.calculate(imu, elh, dlh, update);

        // // optimize
        // fopt_p.states[0] = lf_leg.current();
        // fopt_p.states[1] = rf_leg.current();
        // fopt_p.states[2] = rh_leg.current();
        // fopt_p.states[3] = lh_leg.current();
        // fopt.optimize(u, minf);
        // // std::cout << "minf: " << minf << "\n";
        STATE lf_v, rf_v, rh_v, lh_v;
        lf_v = lf_leg.calculate(imu, elf, dlf, update);
        rf_v = rf_leg.calculate(imu, erf, drf, update);
        rh_v = rh_leg.calculate(imu, erh, drh, update);
        lh_v = lh_leg.calculate(imu, elh, dlh, update);

        estimate_state.row(i).segment(0, 4) = Eigen::Vector4d(lf_v.ground.point(2), rf_v.ground.point(2), rh_v.ground.point(2), lh_v.ground.point(2));
        estimate_state.row(i).segment(4, 4) = 1e-2 * Eigen::Vector4d(df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i));
        estimate_state.row(i).segment(8, 4) = Eigen::Vector4d(lf_v.covariance(0, 0), rf_v.covariance(0, 0), rh_v.covariance(0, 0), lh_v.covariance(0, 0));
        estimate_state.row(i).segment(12, 3) = lf_v.predicted_velocity;
        estimate_state.row(i).segment(15, 3) = rf_v.predicted_velocity;
        estimate_state.row(i).segment(18, 3) = rh_v.predicted_velocity;
        estimate_state.row(i).segment(21, 3) = lh_v.predicted_velocity;

        // estimate_state.row(i).segment(24, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        // estimate_state.row(i).segment(27, 3) = Eigen::Vector3d(u[0], u[1], u[2]);
        counter++;
        // estimate_state.row(i).segment(27, 3) = true_value_estimate;
    }
    std::vector<std::string> cols = {"lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z",
    "lf_.x", "lf_.y", "lf_.z", "rf_.x", "rf_.y", "rf_.z", "rh_.x", "rh_.y", "rh_.z", "lh_.x", "lh_.y", "lh_.z",
    "v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    return 0;
}