#include "../include/forward_propagation.h"
#include "../include/activation_functions.h"

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

    // Calculate Z1
    Eigen::MatrixXf Z1 = W1*X;
    for (int i = 0; i < Z1.cols(); ++i) {
        Z1.col(i) += b1;
    }

    // Get A1 by passing Z1 through activation function
    Eigen::MatrixXf A1 = ReLU(Z1);

    // Calculate Z2
    Eigen::MatrixXf Z2 = W2*A1;
    for (int i = 0; i < Z2.cols(); ++i) {
        Z2.col(i) += b2;
    }

    // Obtain A2 by applying softmax
    Eigen::MatrixXf A2 = softmax(Z2);

    return std::make_tuple(Z1, A1, Z2, A2);
}