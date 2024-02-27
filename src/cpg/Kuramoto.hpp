#ifndef KURAMOTO_CPG_HPP
#define KURAMOTO_CPG_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>

namespace Kuramoto {
    enum TRANSITION_TYPE {
        FEEDBACK = 0x04,
        FAST = 0x02,
        STOP = 0x01,
        DEFAULT = 0x00,
        STOP_FEEDBACK = FEEDBACK | STOP,
        FAST_FEEDBACK = FEEDBACK | FAST,
    };
    class kuramoto_neuron {
        public:
            kuramoto_neuron(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_ = 1e10) ;
            void omega() ;
            void r() ;
            void dydt(double u, double Ky) ;
            void dxdt(double dv,TRANSITION_TYPE t) ;

            void update(double Ky, double f, double dv, double dt, TRANSITION_TYPE t) ;
            void change_param(double alpha_, double beta_, double mu_, double omega_stance_, double omega_swing_, double b_ = 1e10) ;
            double u(double f, double Ky, TRANSITION_TYPE t) ;
            double x;
            double y;
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
    };

    const Eigen::Matrix4d trot_K() ;
    const Eigen::Matrix4d walk_K() ;
    const Eigen::Matrix4d pace_K() ;
    const Eigen::Matrix4d bound_K() ;
}

#endif