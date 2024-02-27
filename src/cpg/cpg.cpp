#include "cpg.hpp"
#include "fstream"
std::mutex mutex_;
double zeta = 0.5;
robot_msg::GAIT gait_type;
void cpg_command_function(double z, robot_msg::GAIT g) {
    mutex_.lock();
    zeta = z;
    gait_type = g;
    std::cout << zeta << "\t" << gait_type << "\n";
    mutex_.unlock();
}

void cb(robot_msg::GaitRequest request, robot_msg::GaitReply &reply) {
    std::cout << "Change CPGs parameters...\n";
    cpg_command_function(request.zeta(), request.type());
    reply.set_zeta(zeta);
    reply.set_type(gait_type);
}

int main(int argc, char* argv[])
{
    Eigen::Matrix<bool, 9, 4> ban_list; ban_list << 0, 0, 1, 1,
                                        0, 1, 1, 0,
                                        1 ,1 ,0 ,0,
                                        1 ,0 ,0 ,1,
                                        1 ,1 ,1 ,0,
                                        0 ,1 ,1 ,1,
                                        1 ,0 ,1 ,1,
                                        1 ,1 ,0 ,1,
                                        1 ,1 ,1 ,1;
    Eigen::Matrix<double, 4, 4> potential; potential << 0, 1, 1, 1,
                                        1, 0, 1, 1,
                                        1 ,1 ,0 ,1,
                                        1 ,1 ,1 ,0;
    Eigen::Vector4d phase; phase << 0.1, -0.2, 0, 0.8;
    double c_gain = 2;
    double p_gain = 60;
    Kuramoto_cpg cpg(ban_list, potential);
    
    double dt = 1.0 / 1000.0;
    cpg(c_gain, p_gain, zeta);
    cpg.phase(phase);

    uint sleep_us = 1000000 / 1000; 
    
    core::NodeHandler nh;
    core::Rate rate(1000);
    
    core::ServiceServer<robot_msg::GaitRequest, robot_msg::GaitReply> &srv = 
    nh.serviceServer<robot_msg::GaitRequest, robot_msg::GaitReply>("gait/command", cb);
    core::Publisher<robot_msg::GaitInfo> &pub = nh.advertise<robot_msg::GaitInfo>("cpg/phase");
    std::ofstream file("cpgs.csv");
    while (1)
    {
        mutex_.lock();
        robot_msg::GaitInfo phase_data;
        cpg(c_gain, p_gain, zeta);
        cpg.phase_generator(gait_type, dt);
        robot_msg::FootPhase *footphase_ptr = phase_data.mutable_phase();
        footphase_ptr->set_zeta(zeta);
        phase_data.set_type(gait_type);
        for (int i = 0; i < 4; i ++) {
            footphase_ptr->mutable_phase()->Add(cpg.phase()(i));
            footphase_ptr->mutable_contact()->Add(cpg.contact()(i));
        }
        mutex_.unlock();
        pub.publish(phase_data);
        file << cpg.phase()(0) << "," << cpg.phase()(1) << "," << cpg.phase()(2) << "," << cpg.phase()(3) << "\n";
        rate.sleep();
    }
    file.close();
    return 0;
}