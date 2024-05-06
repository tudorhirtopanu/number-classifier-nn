#include "../include/helpers.h"
#include <Eigen/Core>
#include <random>

/**
 * @brief Initialize parameters for the neural network.
 *
 * This function initializes the parameters (weights and biases) for the neural network with random values within a specified range.
 * The matrices and vectors are initialized with random values between -0.5 and 0.5.
 *
 * @return A tuple containing the initialized parameters:
 *         - W1: The weight matrix for the first layer
 *         - b1: The bias vector for the first layer
 *         - W2: The weight matrix for the second layer
 *         - b2: The bias vector for the second layer
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> initParams() {
    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize parameters
    Eigen::MatrixXf W1 = Eigen::MatrixXf::Random(10, 784).array() - 0.5;
    Eigen::VectorXf b1 = Eigen::MatrixXf::Random(10, 1).array() - 0.5;
    Eigen::MatrixXf W2 = Eigen::MatrixXf::Random(10, 10).array() - 0.5;
    Eigen::VectorXf b2 = Eigen::MatrixXf::Random(10, 1).array() - 0.5;

    return std::make_tuple(W1, b1, W2, b2);
}

/**
 * @brief One-hot encode labels.
 *
 * This function performs one-hot encoding on integer labels.
 *
 * @param Y The vector of integer labels to be one-hot encoded.
 * @return The one-hot encoded matrix where each column represents a sample
 *         and each row represents a class.
 */
Eigen::MatrixXi oneHotEncode(const Eigen::VectorXi& Y){
    int numSamples = Y.size();
    int numClasses = Y.maxCoeff() + 1;

    // Initialize the one-hot encoded matrix
    Eigen::MatrixXi oneHotY(numClasses, numSamples);
    oneHotY.setZero(); // Initialize with zeros

    // Set the one-hot encoded values
    for (int i = 0; i < numSamples; ++i) {
        oneHotY(Y(i), i) = 1;
    }

    return oneHotY;
}