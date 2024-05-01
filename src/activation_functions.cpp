#include "../include/forward_propagation.h"

#include <Eigen/Dense>

/**
 * @brief Applies the ReLU activation function element-wise to a matrix.
 *
 * This function applies the Rectified Linear Unit activation function element-wise to a given matrix.
 * ReLU sets all negative elements to zero and leaves positive elements unchanged.
 *
 * @param Z The input matrix to which ReLU will be applied.
 * @return A matrix with the same dimensions as the input matrix, where each element is the result of applying ReLU.
 */
Eigen::MatrixXf ReLU(Eigen::MatrixXf Z){
    return Z.array().max(0);
}

/**
 * @brief Applies the softmax function
 *
 * This function computes the softmax function along the columns of a given matrix.
 * Softmax converts raw scores into probabilities for each column independently
 *
 * @param Z The input matrix to which softmax will be applied.
 * @return A matrix with the same dimensions as the input matrix, where each column represents
 *         the softmax probabilities for one data point.
 */
Eigen::MatrixXf softmax(const Eigen::MatrixXf& Z) {
    Eigen::MatrixXf A(Z.rows(), Z.cols());

    // Compute exponential of Z
    Eigen::MatrixXf expZ = Z.array().exp();

    // Compute sum of exponential along each column
    Eigen::VectorXf expZSum = expZ.colwise().sum();

    // Compute softmax for each column
    for (int j = 0; j < Z.cols(); ++j) {
        A.col(j) = expZ.col(j) / expZSum(j);
    }

    return A;
}

/**
 * @brief Computes the derivative of the Rectified Linear Unit function
 *
 * This function calculates the derivative of the ReLU function for each element in the input matrix.
 * The derivative is 1 for positive elements and 0 for non-positive elements.
 *
 * @param Z The input matrix for which the derivative of ReLU will be calculated.
 * @return A matrix with the same dimensions as the input matrix, containing the derivatives of ReLU.
 */
Eigen::MatrixXf ReLU_derivative(Eigen::MatrixXf Z){
    Eigen::MatrixXf derivative(Z.rows(), Z.cols());

    for (int i = 0; i < Z.rows(); ++i) {
        for (int j = 0; j < Z.cols(); ++j) {
            if (Z(i, j) > 0) {
                derivative(i, j) = 1;
            } else {
                derivative(i, j) = 0;
            }
        }
    }

    return derivative;
}