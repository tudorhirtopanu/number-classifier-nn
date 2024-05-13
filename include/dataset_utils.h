#ifndef DATASET_UTILS
#define DATASET_UTILS

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXf readData(const std::string& filename);

Eigen::VectorXi  readLabels(const std::string& filename);

void savePGM(const std::string& filename, const Eigen::MatrixXf& image);

void shuffleDataAndLabels(Eigen::MatrixXf& data, Eigen::VectorXi& labels);

#endif