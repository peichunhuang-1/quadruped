#include "Estimator.hpp"
#include "csv_reader.hpp"
#include <iostream>

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
    int Num = 1;
    VelocityEstimateByLeg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), Eigen::Vector3d(0.2 , 0.08, 0), 0.1, 0.01, 0.005, Num);
    VelocityEstimateByLeg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), Eigen::Vector3d(0.2 ,-0.08, 0), 0.1, 0.01, 0.005, Num);
    VelocityEstimateByLeg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), Eigen::Vector3d(-0.2 ,-0.08, 0), 0.1, 0.01, 0.005, Num);
    VelocityEstimateByLeg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), Eigen::Vector3d(-0.2 ,0.08, 0), 0.1, 0.01, 0.005, Num);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 6);
    BodyEstimation fopt_p(&lf_leg, &rf_leg, &rh_leg, &lh_leg, Num);
    nlopt::opt fopt = Optimizer(&fopt_p);
    int counter = 0;
    std::vector<double> u;
    u.resize(3 * Num, 0);
    double minf = 0;
    for (int i = 1; i < n - 1; i++) {
        bool update = counter % 20 == 0? true: false;
        // std::cout << i << "\n";
        ENCODER_DATA elf = {
            df.iloc("lf.beta", i) + random_number() * 1e-3,
            df.iloc("lf.theta", i) + random_number() * 1e-3,
            df.iloc("lf.beta_d", i) + random_number() * 1e-2,
            df.iloc("lf.theta_d", i) + random_number() * 1e-2,
        };
        ENCODER_DATA erf = {
            df.iloc("rf.beta", i) + random_number() * 1e-3,
            df.iloc("rf.theta", i) + random_number() * 1e-3,
            df.iloc("rf.beta_d", i) + random_number() * 1e-2,
            df.iloc("rf.theta_d", i) + random_number() * 1e-2,
        };
        ENCODER_DATA erh = {
            df.iloc("rh.beta", i) + random_number() * 1e-3,
            df.iloc("rh.theta", i) + random_number() * 1e-3,
            df.iloc("rh.beta_d", i) + random_number() * 1e-2,
            df.iloc("rh.theta_d", i) + random_number() * 1e-2,
        };
        ENCODER_DATA elh = {
            df.iloc("lh.beta", i) + random_number() * 1e-3,
            df.iloc("lh.theta", i) + random_number() * 1e-3,
            df.iloc("lh.beta_d", i) + random_number() * 1e-2,
            df.iloc("lh.theta_d", i) + random_number() * 1e-2,
        };

        double alpha_l = atan2((df.iloc("lh.dist", i) - df.iloc("lf.dist", i)) , 0.4);
        double alpha_r = atan2((df.iloc("rh.dist", i) - df.iloc("rf.dist", i)) , 0.4);
        double alpha_f = atan2((df.iloc("rf.dist", i) - df.iloc("lf.dist", i)) , 0.16);
        double alpha_h = atan2((df.iloc("rh.dist", i) - df.iloc("lh.dist", i)) , 0.16);

        LIDAR_DATA dlf = {
            df.iloc("lf.dist", i) + random_number() * 3e-5,
            alpha_f + random_number() * 1e-3,
            alpha_l + random_number() * 1e-3
        };
        LIDAR_DATA drf = {
            df.iloc("rf.dist", i) + random_number() * 3e-5,
            alpha_f + random_number() * 1e-3,
            alpha_r + random_number() * 1e-3
        };
        LIDAR_DATA drh = {
            df.iloc("rh.dist", i) + random_number() * 3e-5,
            alpha_h + random_number() * 1e-3,
            alpha_r + random_number() * 1e-3
        };
        LIDAR_DATA dlh = {
            df.iloc("lh.dist", i) + random_number() * 3e-5,
            alpha_h + random_number() * 1e-3,
            alpha_l + random_number() * 1e-3
        };
        IMU_DATA imu = {
            Eigen::Vector3d(df.iloc("a.x", i), df.iloc("a.y", i), df.iloc("a.z", i)) + 1e-2 * random_vector<3>() + Eigen::Vector3d(1e-3, 1e-3, 1e-3),
            Eigen::Vector3d(df.iloc("w.x", i), df.iloc("w.y", i), df.iloc("w.z", i)) + 1e-2 * random_vector<3>() + Eigen::Vector3d(1e-3, 1e-3, 1e-3),
            Eigen::Vector4d(df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i), df.iloc("q.w", i)) + 0 * random_vector<4>()
        };
        
        lf_leg.calculate(imu, elf, dlf, update);
        rf_leg.calculate(imu, erf, drf, update);
        rh_leg.calculate(imu, erh, drh, update);
        lh_leg.calculate(imu, elh, dlh, update);
        try {
        fopt.optimize(u, minf);
         }catch (const nlopt::roundoff_limited &ex) {
            // Handle the specific exception
            std::cerr << "Caught nlopt::roundoff_limited exception: " << ex.what() << std::endl;
        }
        fopt_p.assign(u);
        fopt_p.prediction(Eigen::Quaterniond(imu.q).toRotationMatrix());
        fopt_p.calculate_weight();
        estimate_state.row(i).segment(0, 3) = Eigen::Vector3d(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i));
        estimate_state.row(i).segment(3, 3) = Eigen::Vector3d(u[0], u[1], u[2]);
        
        counter ++;
    }
    std::vector<std::string> cols = {"v.x", "v.y", "v.z", "v_.x", "v_.y", "v_.z"};
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
}