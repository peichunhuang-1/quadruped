#include "ParticleFilter.hpp"
#include "csv_reader.hpp"
using namespace estimator;
int main(int argc, char* argv[]) {
    std::string filenumber = std::string(argv[1]);
    std::string filename = "data/data" + filenumber;
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;

    ParticleFilter pf(Eigen::Vector3d(df.iloc("C.x", 0), df.iloc("C.y", 0), df.iloc("C.z", 0)), Eigen::Vector3d(df.iloc("v.x", 0), df.iloc("v.y", 0), df.iloc("v.z", 0)), 
            Eigen::Vector4d(df.iloc("lf.theta", 0),
            df.iloc("lf.beta", 0),
            df.iloc("lf.theta_d", 0),
            df.iloc("lf.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("rf.theta", 0),
            df.iloc("rf.beta", 0),
            df.iloc("rf.theta_d", 0),
            df.iloc("rf.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("rh.theta", 0),
            df.iloc("rh.beta", 0),
            df.iloc("rh.theta_d", 0),
            df.iloc("rh.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("lh.theta", 0),
            df.iloc("lh.beta", 0),
            df.iloc("lh.theta_d", 0),
            df.iloc("lh.beta_d", 0)), 
    300, 0.005);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 36);
    for (int i = 1; i < n - 1; i++) {
        bool update_lidar = ~(i % 100? true: false);
        pf.update(Eigen::Vector3d(df.iloc("a.x", i), df.iloc("a.y", i), df.iloc("a.z", i))
        , Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i)), 
        Eigen::Quaterniond(Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i))));
        if (update_lidar) pf.lidar_measurement(df.iloc("lf.dist", i), df.iloc("rf.dist", i), df.iloc("rh.dist", i), df.iloc("lh.dist", i));
        pf.calculate_weight(Eigen::Vector4d(df.iloc("lf.theta", 0),
            df.iloc("lf.beta", 0),
            df.iloc("lf.theta_d", 0),
            df.iloc("lf.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("rf.theta", 0),
            df.iloc("rf.beta", 0),
            df.iloc("rf.theta_d", 0),
            df.iloc("rf.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("rh.theta", 0),
            df.iloc("rh.beta", 0),
            df.iloc("rh.theta_d", 0),
            df.iloc("rh.beta_d", 0)), 
            Eigen::Vector4d(df.iloc("lh.theta", 0),
            df.iloc("lh.beta", 0),
            df.iloc("lh.theta_d", 0),
            df.iloc("lh.beta_d", 0)), update_lidar);
        auto ps = pf.value();
        pf.resample();
        // std::cout << df.iloc("v.x", i) << "\t" << df.iloc("v.y", i) << "\t" << df.iloc("v.z", i) << "\n";
        // std::cout << ps << "\n";
        estimate_state.row(i).segment(0, 3) = ps.p;
        estimate_state.row(i).segment(3, 3) = ps.v;
        estimate_state.row(i).segment(6, 3) = ps.C_lf;
        estimate_state.row(i).segment(9, 3) = ps.C_rf;
        estimate_state.row(i).segment(12, 3) = ps.C_rh;
        estimate_state.row(i).segment(15, 3) = ps.C_lh;
        estimate_state.row(i).segment(18, 4) = Eigen::Vector4d(ps.contact_expr[3], ps.contact_expr[2], ps.contact_expr[1], ps.contact_expr[0]);
        estimate_state.row(i).segment(22, 3) = ps.G_lf.orient;
    }
    std::vector<std::string> cols = {"C_.x", "C_.y", "C_.z", "v.x", "v.y", "v.z", "lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    return 0;
}