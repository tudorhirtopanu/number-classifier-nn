#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <Eigen/Core>

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> forwardPropagation(Eigen::MatrixXf W1, Eigen::MatrixXf b1, Eigen::MatrixXf W2, Eigen::MatrixXf b2, Eigen::MatrixXf X);

#endif