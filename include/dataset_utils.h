#ifndef DATASET_UTILS
#define DATASET_UTILS

#include <vector>
#include <Eigen/Dense>

Eigen::MatrixXf readData(const std::string& filename);

Eigen::MatrixXi readLabels(const std::string& filename);

#endif