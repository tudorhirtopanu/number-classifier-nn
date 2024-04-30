#ifndef ACTIVATION_FUNCTIONS
#define ACTIVATION_FUNCTIONS

#include <Eigen/Dense>

Eigen::MatrixXf ReLU(Eigen::MatrixXf Z);

Eigen::MatrixXf softmax(const Eigen::MatrixXf& Z);

#endif