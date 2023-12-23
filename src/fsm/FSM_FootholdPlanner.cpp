#include "NodeHandler.h"
#include "FiniteStateMachine.h"
#include "Trajectory.h"
#include "Foothold.h"
#include "robot.pb.h"
#include <fstream>
#define WHEEL_HEIGHT 0.1

std::mutex mutex_;
double zeta = 0.5;
robot_msg::GAIT type;
double steplength = 0;
double liftheight = 0;
double stanceheight = 0;
void gait_command_function(double z, robot_msg::GAIT g, double lift, double stance, double step) {
    mutex_.lock();
    zeta = z;
    type = g;
    steplength = step;
    liftheight = lift;
    stanceheight = stance;
    std::cout << "zeta: " << z << "\n";
    std::cout << "gait: " << g << "\n";
    mutex_.unlock();
}
void cb(robot_msg::GaitRequest request, robot_msg::GaitReply &reply) {
    std::cout << "Change Gait parameters...\n";
    gait_command_function(request.zeta(), request.type(), request.liftheight(), request.stanceheight(), request.steplength());
    reply.set_zeta(zeta);
    reply.set_type(type);
    reply.set_liftheight(liftheight);
    reply.set_stanceheight(stanceheight);
    reply.set_steplength(steplength);
}


int main(int argc, char* argv[])
{
    double freq = 1000;
    FiniteStateMachine fsm(freq);
    double phase[4] = {0, };
    bool contact[4] = {false, false, false, false};
    Point offsets[4] = {Point(0.2, 0.15, 0), Point(0.2, -0.15, 0), Point(-0.2, -0.15, 0), Point(-0.2, 0.15, 0)};
    FootHold planner(freq, WHEEL_HEIGHT, offsets);
    planner.update_footprint(Point(0.2, 0.15, 0), 0);
    planner.update_footprint(Point(0.2, -0.15, 0), 1);
    planner.update_footprint(Point(-0.2, -0.15, 0), 2);
    planner.update_footprint(Point(-0.2, 0.15, 0), 3);
    int ret = 0;

    core::NodeHandler nh;
    core::Rate rate(1000);
    
    core::ServiceServer<robot_msg::GaitRequest, robot_msg::GaitReply> &srv = 
    nh.serviceServer<robot_msg::GaitRequest, robot_msg::GaitReply>("gait/command", cb);
    core::Publisher<robot_msg::GaitInfo> &pub = nh.advertise<robot_msg::GaitInfo>("gait/trajectory");
    while (1) {
        mutex_.lock();
        fsm.update(type, zeta * freq);
        fsm.swing_duration(phase);
        fsm.is_contact(contact);
        Point desire_footprint[4] = {planner.footprint(0), planner.footprint(1), planner.footprint(2), planner.footprint(3)};
        planner.Update(phase, contact, steplength, liftheight, stanceheight, type);
        Point p = planner.centroid(ret);
        robot_msg::GaitInfo gait_data;
        robot_msg::FootPhase *footphase_ptr = gait_data.mutable_phase();
        footphase_ptr->set_zeta(zeta);
        gait_data.set_type(type);
        robot_msg::FootPrint *footprint_ptr = gait_data.mutable_position();
        for (int i = 0; i < 4; i ++) {
            Point f = planner.leg_trajectory(i);
            geometry_msg::Pose footprint;
            footprint.mutable_position()->set_x(f.x);
            footprint.mutable_position()->set_y(f.y);
            footprint.mutable_position()->set_z(f.z);
            footphase_ptr->mutable_phase()->Add(phase[i]);
            footphase_ptr->mutable_contact()->Add(contact[i]);
            footprint_ptr->add_pose()->CopyFrom(footprint);
        }
        gait_data.mutable_centroid()->mutable_position()->set_x(p.x);
        gait_data.mutable_centroid()->mutable_position()->set_y(p.y);
        gait_data.mutable_centroid()->mutable_position()->set_z(p.z);
        mutex_.unlock();
        pub.publish(gait_data);
        rate.sleep();
    }
    return 0;
}