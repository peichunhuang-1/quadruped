#ifndef TRAJECTORY_H
#define TRAJECTORY_H
#include <unordered_map>
#include <mutex>
#include <vector>
#include <iostream>
class Point
{
    public:
    Point(){}
    Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
    double x = 0;
    double y = 0;
    double z = 0;
    double q[4] = {1, 0, 0, 0}; // wxyz
};

class LegTrajectory
{
    public:
    LegTrajectory() {}
    LegTrajectory(Point from, Point to, uint64_t swing_count, uint64_t ref_count, double lift){
        from_ = from;
        to_ = to;
        double tx = to.x; double ty = to.y; double tz = to.z;
        double rx = from.x; double ry = from.y; double rz = from.z;
        uint64_t i;
        for (i = 0; i < swing_count; i++)
        {
            double ratio = (double) i / (double)swing_count;
            double x = (tx - rx) * ratio + rx;
            double y = (ty - ry) * ratio + ry;
            double z = (tz - rz) * ratio + 4 * lift * (1.0 - ratio) * ratio + rz;
            trajectories_[i + ref_count] = Point(x, y, z);
        }
        max_count = i + ref_count - 1;
        ref = ref_count;
    }
    Point position(uint64_t count)
    {
        if (count > max_count)
        {
            return to_;
        }
        else if (count < ref)
        {
            return from_;
        }
        else
        {
            return trajectories_[count];
        }
    }
    private:
    std::unordered_map<uint64_t, Point> trajectories_;
    uint64_t max_count = 1;
    uint64_t ref = 0;
    Point from_;
    Point to_;
};

class Trajectory
{
    public:
    enum TYPE{
        COUNTED = 0,
        COUNT_EARLY = 1,
        COUNT_OUT = 2
    };
    Trajectory(double f, double dt, Point target, uint64_t reference_count, Point reference = Point()) 
    : freq_(f), reference_count_(reference_count)
    {
        uint64_t count = f * dt;
        if (count == 0) return;
        double tx = target.x; double ty = target.y; double tz = target.z;
        double rx = reference.x; double ry = reference.y; double rz = reference.z;
        uint64_t i;
        for (i = 0; i < count; i++)
        {
            double ratio = (double) i / (double)count;
            double x = (tx - rx) * ratio + rx;
            double y = (ty - ry) * ratio + ry;
            double z = (tz - rz) * ratio + rz;
            trajectories_[i] = Point(x, y, z);
        }
        max_count = reference_count + i - 1;
    }
    Point trajectory_at_count(uint64_t count, int &ret) 
    {
        if (count > max_count) ret = COUNT_OUT;
        else if (count < reference_count_) ret = COUNT_EARLY;
        else ret = COUNTED;
        return trajectories_[count-reference_count_];
    }
    uint64_t end_count() {return max_count;}
    private:
    double freq_ = 1000.0;
    std::unordered_map<uint64_t, Point> trajectories_;
    uint64_t reference_count_ = 0;
    uint64_t max_count = 1;
};

class TrajectoriesPool
{
    public:
        TrajectoriesPool() {}
        void add(double f, double dt, Point target, uint64_t reference_count, Point reference)
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            if (! pool.empty()) {
                if (reference_count < pool.back().end_count())
                {
                    reference_count = pool.back().end_count();
                    int ret;
                    reference = pool.back().trajectory_at_count(reference_count, ret);
                }
            }
            pool.push_back(Trajectory(f, dt, target, reference_count, reference));
        }
        Point point_at_count(uint64_t count, int &ret)
        {
            Point traj_point;
            int ret_ = 0;
            {
                std::lock_guard<std::mutex> lock_guard(mutex_);
                if (pool.empty()) {
                    ret = Trajectory::COUNT_OUT;
                    return traj_point_;
                }
                traj_point = pool.front().trajectory_at_count(count, ret_);
                if (ret_ == Trajectory::COUNT_OUT) {
                    pool.erase(pool.begin());
                }
                if (ret_ == Trajectory::COUNT_EARLY){
                    return traj_point_;
                }
            }
            if (ret_ == Trajectory::COUNT_OUT)
            {
                return point_at_count(count, ret);
            }
            else{
                traj_point_ = traj_point;
                return traj_point;
            }
        }
        void set_origin(Point origin)
        {
            traj_point_ = origin;
        }
        void clear()
        {
            std::lock_guard<std::mutex> lock_guard(mutex_);
            pool.clear();
        }
    private:
        std::mutex mutex_;
        std::vector<Trajectory> pool;
        Point traj_point_;

};

#endif