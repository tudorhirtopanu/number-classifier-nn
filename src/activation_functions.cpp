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