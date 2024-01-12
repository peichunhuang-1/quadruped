#include <iostream>
#include "csv_reader.hpp"
#include "ObservationModel.hpp"
#include "Optimizer.hpp"
int main(int argc, char* argv[]) {
    std::string filenumber = std::string(argv[1]);
    std::string filename = "data/data" + filenumber;
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;
    LegVelocityEstimation lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01, 0.005);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 30);
    Eigen::Vector3d true_value_estimate;
    int counter = 0;
    VelocityOptimizer fopt_p;
    nlopt::opt fopt = Optimizer(&fopt_p);
    std::vector<double> u {0, 0, 0};
    double minf = 0;

    for (int i = 1; i < n - 1; i++) {
        bool update = counter % 5 == 0? true: false;
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
        lf_leg.observation(imu, elf, dlf, update);
        rf_leg.observation(imu, erf, drf, update);
        rh_leg.observation(imu, erh, drh, update);
        lh_leg.observation(imu, elh, dlh, update);

        // Eigen::Vector3d a_in_world = (Eigen::Quaterniond(imu.q).toRotationMatrix() * (imu.a)).cwiseAbs() + Eigen::Vector3d(1e-3, 1e-3, 1e-3);
        // Eigen::Vector3d v_last(u.data());
        // Eigen::Vector3d lb = v_last - a_in_world * 0.005 - Eigen::Vector3d(1e-3, 1e-3, 1e-3);
        // Eigen::Vector3d ub = v_last + a_in_world * 0.005 + Eigen::Vector3d(1e-3, 1e-3, 1e-3);

        // fopt.set_lower_bounds(std::vector<double>(lb.data(), lb.data()+3));
        // fopt.set_upper_bounds(std::vector<double>(ub.data(), ub.data()+3));

        // optimize
        fopt_p.states[0] = lf_leg.current();
        fopt_p.states[1] = rf_leg.current();
        fopt_p.states[2] = rh_leg.current();
        fopt_p.states[3] = lh_leg.current();
        fopt.optimize(u, minf);
        // std::cout << "minf: " << minf << "\n";
        Eigen::Vector3d lf_v, rf_v, rh_v, lh_v;
        lf_v = fopt_p.states[0].observed_velocity;
        rf_v = fopt_p.states[1].observed_velocity;
        rh_v = fopt_p.states[2].observed_velocity;
        lh_v = fopt_p.states[3].observed_velocity;

        estimate_state.row(i).segment(0, 4) = Eigen::Vector4d(lf_leg.current().covariance(0, 0), rf_leg.current().covariance(0, 0), rh_leg.current().covariance(0, 0), lh_leg.current().covariance(0, 0));
        estimate_state.row(i).segment(4, 4) = Eigen::Vector4d(df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i));
        estimate_state.row(i).segment(8, 4) = Eigen::Vector4d(fopt_p.states[0].weight, fopt_p.states[1].weight, fopt_p.states[2].weight, fopt_p.states[3].weight);
        estimate_state.row(i).segment(12, 3) = lf_v;
        estimate_state.row(i).segment(15, 3) = rf_v;
        estimate_state.row(i).segment(18, 3) = rh_v;
        estimate_state.row(i).segment(21, 3) = lh_v;

        estimate_state.row(i).segment(24, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        estimate_state.row(i).segment(27, 3) = Eigen::Vector3d(u[0], u[1], u[2]);
        counter++;
        // estimate_state.row(i).segment(27, 3) = true_value_estimate;
    }
    std::vector<std::string> cols = {"lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z",
    "lf_.x", "lf_.y", "lf_.z", "rf_.x", "rf_.y", "rf_.z", "rh_.x", "rh_.y", "rh_.z", "lh_.x", "lh_.y", "lh_.z",
    "v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    std::cout << estimate_state.block(0, 0, 10, 1) << "\n";
    return 0;
}