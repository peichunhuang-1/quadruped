#ifndef CPG_HPP
#define CPG_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include "kinematic/Leg.hpp"

namespace Kuramoto {
    double mu_func(double phi, double mu_max = 1., double d = 0.5) ;
    class MCKsystem {
        public:
            MCKsystem(double M, double C, double K, double x0=0, double ulimit=1, double llimit=-1.) ;
            void F(double u, double dt = 0.001) ;
            double x;
            double x_d;
            double x_dd;
        private:
            double m;
            double c;
            double k;
            double zero_point;
            double u;
            double l;
    };
    class kuramoto_neuron {
        enum type {
            DEFAULT = 0,
            FAST = 1,
            STOP = 2,
        };
        public:
            kuramoto_neuron(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_ = 1e10) ;
            void omega() ;
            void r() ;
            void dydt(double Ky, double F = 0) ;
            void dxdt() ;
            
            void update(double Ky, double F, double dt) ;
            void change_param(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_ = 1e10) ;
            double x;
            double y;
            double phi;
            double mu_d;
            std::pair<double, double> output(double beta_const, double lift, double dig, Leg &leg, double dt) ;
            MCKsystem* beta_system;
            MCKsystem* stance_height_system;
        private:
            double omega_stance;
            double omega_swing;

            double x_dot;
            double y_dot;
            double radius;
            double w;

            double alpha;
            double beta;
            double mu;
            double b;
            type T(double F);
    };

    const Eigen::Matrix4d trot_K() ;
    const Eigen::Matrix4d walk_K() ;
    const Eigen::Matrix4d pace_K() ;
    const Eigen::Matrix4d bound_K() ;
}

#endif