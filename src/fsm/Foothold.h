#ifndef FOOTHOLD_H
#define FOOTHOLD_H
#include "Trajectory.h"
#include "robot.pb.h"
#include "geometry.pb.h"
class FootHold
{
    private:
        Point offset[4];
        Point footprint_[4];
        bool state[4] = {false, false, false, false};
        double switch_phase[4] = {0, 0, 0, 0};
        TrajectoriesPool cm_trajectory;
        LegTrajectory leg_trajectories[4];
        double freq = 1000.0;
        uint64_t count = 0;
        int stance_rest = 0;
    public:
        FootHold(double f, double wheel_height, Point *offsets = nullptr) : freq(f) {
            cm_trajectory.set_origin(Point(0, 0, wheel_height));
            if (offsets != nullptr) {
                for (int i = 0; i < 4; i++) offset[i] = *(offsets+i);
            }
        }
        void update_footprint(Point footprint_i, int i) {
            footprint_[i] = footprint_i;
        }
        Point centroid(int &ret) {
            return cm_trajectory.point_at_count(count, ret);
        }
        Point footprint(int i) {
            return footprint_[i];
        }
        Point leg_trajectory(int i) {
            if (state[i]) {
                return leg_trajectories[i].position(count);
            }
            else {
                return footprint_[i];
            }
        }
        void Update(double phases[4], bool contacts[4], double steplength, double liftheight, double stanceheight, robot_msg::GAIT type, 
        Point *footprint_desired = nullptr, Point *feedbacks = nullptr, bool feedback = false, double heading = 0, double terrain_height = 0) {
            double total_phase = 0;
            uint8_t flag = 0;
            uint8_t bit_count = 1;
            for (uint8_t i = 0; i < 4; i ++) {
                bit_count *= 2;
                double phase = phases[i];
                if (contacts[i] && !state[i]) {
                    switch_phase[i] = phase;
                    total_phase += phase;
                    flag = (flag | bit_count);
                }
                state[i] = contacts[i];
            }
            int ret = 0;
            Point centro_point = this->centroid(ret);
            switch(type)
            {
                case robot_msg::STOP :
                {

                }
                break;
                case robot_msg::STANCE :
                {
                    if (stance_rest < 0) {
                        stance_rest = freq;
                        cm_trajectory.add(freq, 1.0, Point(centro_point.x, centro_point.y, terrain_height + stanceheight), count, cm_trajectory.point_at_count(count, ret));
                    }
                }
                break;
                case robot_msg::TROT :
                {
                    if (flag) {
                        bit_count = 1;
                        for (uint8_t i = 0; i < 4; i ++) {
                            bit_count *= 2;
                            double phase = phases[i];
                            if (bit_count & flag) {
                                Point cp = footprint_[i];
                                Point centro_later = Point(centro_point.x + cos(heading) * steplength * 0.5, centro_point.y + sin(heading) * steplength * 0.5, terrain_height + stanceheight);
                                double dT = total_phase / freq ;
                                double step_trot = abs(2 * (1 - exp(dT * sqrt(9.81 / stanceheight))) / (1 - exp(2 * dT * sqrt(9.81 / stanceheight))) - 1) * steplength * 0.5 / dT * sqrt(stanceheight / 9.81);
                                footprint_[i] = Point(centro_later.x + offset[i].x * cos(heading) + step_trot * cos(heading), centro_later.y + offset[i].y * sin(heading) + step_trot * sin(heading), terrain_height);
                                leg_trajectories[i] = LegTrajectory(cp, footprint_[i], phase, count, liftheight);
                                if (i >= 2) {
                                    cm_trajectory.add(freq, total_phase / freq, centro_later, count, cm_trajectory.point_at_count(count, ret));
                                }
                            }
                        }
                    }
                }
                break;
                case robot_msg::WALK :
                {
                    if (flag) {
                        cm_trajectory.add(freq, total_phase / freq, Point(centro_point.x + cos(heading) * steplength * 0.25, centro_point.y + sin(heading) * steplength * 0.25, terrain_height + stanceheight), count, centro_point);
                        bit_count = 1;
                        for (uint8_t i = 0; i < 4; i ++) {
                            bit_count *= 2;
                            double phase = phases[i];
                            if (bit_count & flag) {
                                Point cp = footprint_[i];
                                Point centro_later = Point(centro_point.x + cos(heading) * steplength * 0.25, centro_point.y + sin(heading) * steplength * 0.25, terrain_height + stanceheight);
                                footprint_[i] = Point(centro_later.x + offset[i].x * cos(heading) + 0.375 * steplength * cos(heading), centro_later.y + offset[i].y * sin(heading) + 0.375 * steplength * sin(heading), terrain_height);
                                leg_trajectories[i] = LegTrajectory(cp, footprint_[i], phase, count, liftheight);
                            }
                        }
                    }
                }
                break;
                case robot_msg::CONGNITIVE :
                {
                    double cx = 0;
                    double cy = 0;
                    double cz = 0;
                    for (uint8_t i = 0; i < 4; i ++) {
                        bit_count *= 2;
                        double phase = phases[i];
                        if (bit_count & flag) {
                            Point cp = footprint_[i];
                            footprint_[i] = Point(cp.x + cos(heading) * steplength, cp.y + sin(heading) * steplength, terrain_height);
                            leg_trajectories[i] = LegTrajectory(cp, footprint_[i], phase, count, liftheight);
                        }
                        cx += footprint_[i].x;
                        cy += footprint_[i].y;
                        cz += footprint_[i].z;
                    }
                    cx *= 0.25;
                    cy *= 0.25;
                    cz *= 0.25;
                    cm_trajectory.add(freq, total_phase / freq, Point(cx, cy, cz + stanceheight), count, centro_point);
                }
                break;
                default:
                {
                    
                }
                break;
            }
            count ++;
            stance_rest --;
        }
};

#endif