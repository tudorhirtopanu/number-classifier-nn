#ifndef HELPERS
#define HELPERS

#include <Eigen/Core>

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> initParams();

Eigen::MatrixXi oneHotEncode(const Eigen::VectorXi& Y);

Eigen::VectorXi getPredictions(const Eigen::MatrixXf& A2);

double getAccuracy(const Eigen::VectorXi& predictions, const Eigen::VectorXi& Y);

int findMaxIndex(const Eigen::MatrixXf& matrix);

#endif