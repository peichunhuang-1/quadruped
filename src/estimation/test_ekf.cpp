#include "EKF.hpp"
#include <iostream>
#include "csv_reader.hpp"
#include <random>

using namespace estimation_model;

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
    int j = 10;
    double dt = 0.005;
    Leg lf_leg(Eigen::Vector3d(0.2, 0.15, 0), 0.1, 0.01);
    Leg rf_leg(Eigen::Vector3d(0.2, -0.15, 0), 0.1, 0.01);
    Leg rh_leg(Eigen::Vector3d(-0.2, -0.15, 0), 0.1, 0.01);
    Leg lh_leg(Eigen::Vector3d(-0.2, 0.15, 0), 0.1, 0.01);
    Eigen::MatrixXd estimate_state = Eigen::MatrixXd::Zero(n, 32);

    Eigen::Vector3d a(df.iloc("a.x", 0), df.iloc("a.y", 0), df.iloc("a.z", 0));
    Eigen::Quaterniond q(df.iloc("q.w", 0), df.iloc("q.x", 0), df.iloc("q.y", 0), df.iloc("q.z", 0));
    Eigen::Vector<double, 5> contact_vector_lf(df.iloc("lf.theta", 0), df.iloc("lf.beta", 0), df.iloc("lf.beta_d", 0), df.iloc("w.y", 0), df.iloc("lf.theta_d", 0));
    Eigen::Vector<double, 5> contact_vector_rf(df.iloc("rf.theta", 0), df.iloc("rf.beta", 0), df.iloc("rf.beta_d", 0), df.iloc("w.y", 0), df.iloc("rf.theta_d", 0));
    Eigen::Vector<double, 5> contact_vector_rh(df.iloc("rh.theta", 0), df.iloc("rh.beta", 0), df.iloc("rh.beta_d", 0), df.iloc("w.y", 0), df.iloc("rh.theta_d", 0));
    Eigen::Vector<double, 5> contact_vector_lh(df.iloc("lh.theta", 0), df.iloc("lh.beta", 0), df.iloc("lh.beta_d", 0), df.iloc("w.y", 0), df.iloc("lh.theta_d", 0));
    Eigen::Vector3d v_init(df.iloc("v.x", 0), df.iloc("v.y", 0), df.iloc("v.z", 0)) ;
    Eigen::MatrixXd x;
    Eigen::Matrix3d R = q.toRotationMatrix();
    x.resize(j, 3);
    for (int i = 0; i < j; i++) x.row(i) = v_init;
    EKF filter(j, dt) ;
    filter.init(x);
    U input(j, a, R) ;
    Z observed_lf(j+1, contact_vector_lf, R, 0);
    Z observed_rf(j+1, contact_vector_rf, R, 0);
    Z observed_rh(j+1, contact_vector_rh, R, 0);
    Z observed_lh(j+1, contact_vector_lh, R, 0);

    std::deque<Eigen::Vector3d> P;
    for (int i = 0; i < j + 1; i++) P.push_back(Eigen::Vector3d(df.iloc("C.x", 0), df.iloc("C.y", 0), df.iloc("C.z", 0)));

    std::deque<std::tuple<bool, bool, bool, bool> > contacts;
    for (int i = 0; i < j + 1; i++) 
        contacts.push_back(std::tuple<bool, bool, bool, bool> {0, 0, 0, 0});
    int counter = 0;
    for (int i = 2; i < n; i++) {
        counter ++ ;
        Eigen::Vector3d a(df.iloc("a.x", i-2), df.iloc("a.y", i-2), df.iloc("a.z", i-2)); a += 1e-2 * random_vector<3>() + Eigen::Vector3d(1e-3, 1e-3, 1e-3);
        Eigen::Quaterniond qk_2(df.iloc("q.w", i-2), df.iloc("q.x", i-2), df.iloc("q.y", i-2), df.iloc("q.z", i-2));
        Eigen::Matrix3d R_k2 = q.toRotationMatrix();
        Eigen::Quaterniond q(df.iloc("q.w", i), df.iloc("q.x", i), df.iloc("q.y", i), df.iloc("q.z", i));
        Eigen::Vector<double, 5> contact_vector_lf(df.iloc("lf.theta", i), df.iloc("lf.beta", i), df.iloc("lf.beta_d", i), df.iloc("w.y", i), df.iloc("lf.theta_d", i));
        Eigen::Vector<double, 5> contact_vector_rf(df.iloc("rf.theta", i), df.iloc("rf.beta", i), df.iloc("rf.beta_d", i), df.iloc("w.y", i), df.iloc("rf.theta_d", i));
        Eigen::Vector<double, 5> contact_vector_rh(df.iloc("rh.theta", i), df.iloc("rh.beta", i), df.iloc("rh.beta_d", i), df.iloc("w.y", i), df.iloc("rh.theta_d", i));
        Eigen::Vector<double, 5> contact_vector_lh(df.iloc("lh.theta", i), df.iloc("lh.beta", i), df.iloc("lh.beta_d", i), df.iloc("w.y", i), df.iloc("lh.theta_d", i));
        Eigen::Vector3d v(df.iloc("v.x", i), df.iloc("v.y", i), df.iloc("v.z", i)) ;
        Eigen::MatrixXd x;
        Eigen::MatrixXd z;
        Eigen::MatrixXd Q;
        P.push_back(Eigen::Vector3d(df.iloc("C.x", i), df.iloc("C.y", i), df.iloc("C.z", i)));
        P.pop_front();
        z.resize(4, 3);
        Q.resize(4, 4);
        Eigen::Matrix3d R = q.toRotationMatrix();
        std::vector<bool> contact_metrice = {false, false, false, false};
        contacts.push_back(std::tuple<bool, bool, bool, bool> {df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i)});
        contacts.pop_front();
        
        // for (int k = 0; k < j + 1; k++) {
        //     if(std::get<0>(contacts[k])) contact_metrice[0] = false;
        //     if(std::get<1>(contacts[k])) contact_metrice[1] = false;
        //     if(std::get<2>(contacts[k])) contact_metrice[2] = false;
        //     if(std::get<3>(contacts[k])) contact_metrice[3] = false;
        // }

        input.push_data(a, R_k2);
        double alpha;
        if (counter % 4) alpha = -100;
        else alpha = 0;
        observed_lf.push_data(contact_vector_lf, R, dt, alpha);
        observed_rf.push_data(contact_vector_rf, R, dt, alpha);
        observed_rh.push_data(contact_vector_rh, R, dt, alpha);
        observed_lh.push_data(contact_vector_lh, R, dt, alpha);

        z.row(0) = observed_lf.z(lf_leg, dt);
        z.row(1) = observed_rf.z(rf_leg, dt);
        z.row(2) = observed_rh.z(rh_leg, dt);
        z.row(3) = observed_lh.z(lh_leg, dt);

        filter.predict(input.u(dt), input.noise());

        int kick_outlier_index = 0;
        double min = 1000;
        for (int k = 0; k < 4; k++) {
            double diff = (z.row(k) - filter.predicted_m().row(k)).norm();
            if (diff < min) {
                min = diff;
                kick_outlier_index = k;
            }
        }
        contact_metrice[kick_outlier_index] = true;

        Q(0, 0) = contact_metrice[0] ? observed_lf.noise(): 1;
        Q(1, 1) = contact_metrice[1] ? observed_rf.noise(): 1;
        Q(2, 2) = contact_metrice[2] ? observed_rh.noise(): 1;
        Q(3, 3) = contact_metrice[3] ? observed_lh.noise(): 1;


        filter.valid(z, Q);
        x = filter.state();
        estimate_state.row(i).segment(0, 3) = x.row(j - 1);
        estimate_state.row(i).segment(3, 3) = v;
        estimate_state.row(i).segment(6, 3) = z.row(0);
        estimate_state.row(i).segment(9, 3) = z.row(1);
        estimate_state.row(i).segment(12, 3) = z.row(2);
        estimate_state.row(i).segment(15, 3) = z.row(3);
        estimate_state.row(i).segment(18, 3) = (P.back() - P.front());
        estimate_state.row(i).segment(21, 3) = filter.predicted_m().row(0);
        estimate_state.row(i).segment(24, 4) = Eigen::Vector4d(contact_metrice[0], contact_metrice[1], contact_metrice[2], contact_metrice[3]);
        estimate_state.row(i).segment(28, 4) = Eigen::Vector4d(df.iloc("lf.contact", i), df.iloc("rf.contact", i), df.iloc("rh.contact", i), df.iloc("lh.contact", i));
    }
    std::vector<std::string> cols = {
        "v_.x", "v_.y", "v_.z", 
        "v.x", "v.y", "v.z", 
        "zLF.x", "zLF.y", "zLF.z", 
        "zRF.x", "zRF.y", "zRF.z", 
        "zRH.x", "zRH.y", "zRH.z", 
        "zLH.x", "zLH.y", "zLH.z", 
        "zP.x", "zP.y", "zP.z", 
        "zp.x", "zp.y", "zp.z", 
        "lf.contact","rf.contact","rh.contact","lh.contact",
        "lf.c","rf.c","rh.c","lh.c"
    };
    DataProcessor::write_csv(estimate_state, "out_"+filename+"_"+".csv", cols);
    return 0;
}