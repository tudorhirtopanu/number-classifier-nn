#include "../include/forward_propagation.h"

#include <Eigen/Core>

/**
 * @brief Forward propagation for neural network
 *
 * This function performs forward propagation for the neural network,
 * computing the activations of each layer based on the input data
 * and the parameters (weights and biases) of the network.
 *
 * @param W1 Weight matrix for the first layer.
 * @param b1 Bias vector for the first layer.
 * @param W2 Weight matrix for the second layer.
 * @param b2 Bias vector for the second layer.
 * @param X Input data matrix.
 *
 * @return A tuple containing the activations and outputs of each layer:
 *         - The activation of the first hidden layer.
 *         - The output of the first hidden layer after activation.
 *         - The activation of the output layer.
 *         - The output of the output layer after activation
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> forwardPropagation(Eigen::MatrixXf W1, Eigen::MatrixXf b1, Eigen::MatrixXf W2, Eigen::MatrixXf b2, Eigen::MatrixXf X){

}