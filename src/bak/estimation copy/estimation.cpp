#include <iostream>
#include "csv_reader.hpp"
#include "ObservationModel.hpp"
#include "Optimizer.hpp"
#include <random>
template<size_t n>
Eigen::Vector<double, n> random_vector() {
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::uniform_real_distribution<double> dis(-1, 1);
    Eigen::Vector<double, n> V = Eigen::Vector<double, n>().NullaryExpr([&](){return dis(gen);});
    return V;
}

double random_number() {
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could also set a seed
    std::uniform_real_distribution<double> dis(-1, 1);
    return dis(gen);
}

int main(int argc, char* argv[]) {
    std::string filenumber = std::string(argv[1]);
    std::string filename = "data/data" + filenumber;
    DataProcessor::DataFrame df =  DataProcessor::read_csv(filename+".csv");
    int n = df.row;
    LegVelocityEstimation lf_leg(Eigen::Vector3d(0.2, 0.15, 0), Eigen::Vector3d(0.2 , 0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rf_leg(Eigen::Vector3d(0.2, -0.15, 0), Eigen::Vector3d(0.2 ,-0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), Eigen::Vector3d(-0.2 ,-0.08, 0), 0.1, 0.01, 0.005);
    LegVelocityEstimation lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), Eigen::Vector3d(-0.2 ,0.08, 0), 0.1, 0.01, 0.005);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 36);
    Eigen::Vector3d true_value_estimate;
    int counter = 0;
    BodyEstimation fopt_p(&lf_leg, &rf_leg, &rh_leg, &lh_leg, Eigen::Vector3d(df.iloc("v.x", 0), df.iloc("v.y", 0), df.iloc("v.z", 0)));
    fopt_p.current_position = Eigen::Vector3d(df.iloc("C.x", 0), df.iloc("C.y", 0), df.iloc("C.z", 0));
    nlopt::opt fopt = Optimizer(&fopt_p);
    std::vector<double> u {0, 0, 0, .5, .5, .5, .5};
    double minf = 0;
    ContactMap cm;
    for (int i = 1; i < n - 1; i++) {
        bool update = counter % 20 == 0? true: false;
        // std::cout << i << "\n";
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

        double alpha_l = atan2((df.iloc("lf.dist", i) + 1e-3 * random_number() - df.iloc("lh.dist", i) - 1e-3 * random_number()) , 0.4);
        double alpha_r = atan2((df.iloc("rf.dist", i) + 1e-3 * random_number() - df.iloc("rh.dist", i) - 1e-3 * random_number()) , 0.4);
        DST_DATA dlf = {
            df.iloc("lf.dist", i) + 1e-3 * random_number(),
            alpha_l
        };
        DST_DATA drf = {
            df.iloc("rf.dist", i) + 1e-3 * random_number(),
            alpha_r
        };
        DST_DATA drh = {
            df.iloc("rh.dist", i) + 1e-3 * random_number(),
            alpha_r
        };
        DST_DATA dlh = {
            df.iloc("lh.dist", i) + 1e-3 * random_number(),
            alpha_l
        };
        IMU_DATA imu = {
            Eigen::Vector3d(df.iloc("a.x", i), df.iloc("a.y", i), df.iloc("a.z", i)) + 3e-2 * random_vector<3>(),
            Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i)) + 2e-2 * random_vector<3>(),
            Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i)) + 3e-4 * random_vector<4>()
        };

        
        STATE lf_v, rf_v, rh_v, lh_v;
        lf_v = lf_leg.calculate(imu, elf, dlf, update);
        rf_v = rf_leg.calculate(imu, erf, drf, update);
        rh_v = rh_leg.calculate(imu, erh, drh, update);
        lh_v = lh_leg.calculate(imu, elh, dlh, update);
        // optimize

        try {
        fopt.optimize(u, minf);
         }catch (const nlopt::roundoff_limited &ex) {
            // Handle the specific exception
            std::cerr << "Caught nlopt::roundoff_limited exception: " << ex.what() << std::endl;
        }
        Eigen::Quaterniond quat = Eigen::Quaterniond(Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i)));
        Eigen::Matrix3d rot = quat.toRotationMatrix();
        lf_leg.leg.Calculate(df.iloc("lf.theta", i), df.iloc("lf.theta_d", i), 0, df.iloc("lf.beta", i), df.iloc("lf.beta_d", i), 0);
        lf_leg.leg.PointContact(cm.lookup(df.iloc("lf.theta", i), df.iloc("lf.beta", i) + alpha_l), alpha_l);
        lf_leg.leg.PointVelocity(rot.transpose() * Eigen::Vector3d(u[0], u[1], u[2]), Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i)),
            cm.lookup(df.iloc("lf.theta", i), df.iloc("lf.beta", i) + alpha_l), alpha_l,
            false
        );

        fopt_p.prediction();
        fopt_p.update_weight(update);
        estimate_state.row(i).segment(0, 4) = Eigen::Vector4d(u[3], u[4], u[5], u[6]);
        estimate_state.row(i).segment(4, 4) = Eigen::Vector4d(df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i));
        estimate_state.row(i).segment(8, 4) = fopt_p.weights;
        estimate_state.row(i).segment(12, 3) = lf_leg.g.point;
        estimate_state.row(i).segment(15, 3) = rf_leg.g.point;
        estimate_state.row(i).segment(18, 3) = rh_leg.g.point;
        estimate_state.row(i).segment(21, 3) = lh_leg.g.point;

        estimate_state.row(i).segment(24, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        estimate_state.row(i).segment(27, 3) = Eigen::Vector3d(u[0], u[1], u[2]);
        estimate_state.row(i).segment(30, 3) = Eigen::Vector3d(df.iloc("C.x", i), df.iloc("C.y", i), df.iloc("C.z", i));
        estimate_state.row(i).segment(33, 3) = fopt_p.current_position;
        counter++;
        // estimate_state.row(i).segment(27, 3) = fopt_p.current_position;
    }
    std::vector<std::string> cols = {"lf.x", "lf.y", "lf.z", "rf.x", "rf.y", "rf.z", "rh.x", "rh.y", "rh.z", "lh.x", "lh.y", "lh.z",
    "lf_.x", "lf_.y", "lf_.z", "rf_.x", "rf_.y", "rf_.z", "rh_.x", "rh_.y", "rh_.z", "lh_.x", "lh_.y", "lh_.z",
    "v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z", "C.x", "C.y", "C.z", "C_.x", "C_.y", "C_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    return 0;
}