#ifndef PARAMETER_HANDLER
#define PARAMETER_HANDLER

void saveParameters(const Eigen::MatrixXf& W1, const Eigen::VectorXf& b1, const Eigen::MatrixXf& W2, const Eigen::VectorXf& b2, const std::string& filename);

std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::MatrixXf, Eigen::VectorXf>loadParameters(const std::string& filename);

#endif