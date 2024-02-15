#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <map>
#include <stdexcept>
#include "ObservationModel.hpp"
// C.x	C.y	C.z	v.x	v.y	v.z	a.x	a.y	a.z	w.x	w.y	w.z	q.w	q.x	q.y	q.z	lf.theta	
// lf.beta lf.theta_d lf.beta_d rf.theta rf.beta rf.theta_d	rf.beta_d rh.theta rh.beta rh.theta_d 
// rh.beta_d lh.theta lh.beta lh.theta_d lh.beta_d	
// lf.contact	rf.contact	rh.contact	lh.contact	lf.dist	rf.dist	rh.dist	lh.dist

void filing(std::ofstream &file, Eigen::Vector3d a, Eigen::Vector3d w, Eigen::Vector4d q, 
ENCODER_DATA lf, ENCODER_DATA rf, ENCODER_DATA rh, ENCODER_DATA lh, 
double lfd, double rfd, double rhd, double lhd) {
    file << 0 << "," << 0 << "," << 0 << ",";
    file << 0 << "," << 0 << "," << 0 << ",";
    file << a(0) << "," << a(1) << "," << a(2) << ",";
    file << w(0) << "," << w(1) << "," << w(2) << ",";
    file << q(3) << "," << q(0) << "," << q(1) << "," << q(2) << ",";
    file << lf.theta << "," << lf.beta << "," << lf.theta_d << "," << lf.beta_d << ",";
    file << rf.theta << "," << rf.beta << "," << rf.theta_d << "," << rf.beta_d << ",";
    file << rh.theta << "," << rh.beta << "," << rh.theta_d << "," << rh.beta_d << ",";
    file << lh.theta << "," << lh.beta << "," << lh.theta_d << "," << lh.beta_d << ",";
    file << 0 << "," << 0 << "," << 0 << "," << 0 << "," ;
    file << lfd << "," << rfd << "," << rhd << "," << lhd << "\n" ;

}