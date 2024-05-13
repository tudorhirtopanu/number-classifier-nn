#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <Eigen/Core>

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> forwardPropagation( const Eigen::MatrixXf& W1, const Eigen::MatrixXf& b1,const Eigen::MatrixXf& W2,const Eigen::MatrixXf& b2,const Eigen::MatrixXf& X);

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> backwardPropagation(Eigen::MatrixXf Z1, Eigen::MatrixXf A1, Eigen::MatrixXf Z2, Eigen::MatrixXf A2,Eigen::MatrixXf W1, Eigen::MatrixXf W2, Eigen::MatrixXf X, Eigen::VectorXi Y);

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> updateParameters(Eigen::MatrixXf W1,Eigen::MatrixXf b1,Eigen::MatrixXf W2, Eigen::MatrixXf b2,Eigen::MatrixXf dW1,Eigen::MatrixXf db1, Eigen::MatrixXf dW2,Eigen::MatrixXf db2,float alpha);

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> gradientDescent(Eigen::MatrixXf X,Eigen::VectorXi Y,Eigen::MatrixXf valX,Eigen::VectorXi valY, float alpha, int iterations);

Eigen::MatrixXf runImageThroughNetwork(const Eigen::MatrixXf& image, const Eigen::MatrixXf& W1, const Eigen::MatrixXf& b1, const Eigen::MatrixXf& W2, const Eigen::MatrixXf& b2);

#endif