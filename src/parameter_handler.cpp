#include <iostream>
#include <fstream>
#include "Eigen/Dense"

#include "../include/parameter_handler.h"

/**
 * @brief Save neural network parameters to a file
 *
 * This function saves the parameters of a neural network to a binary file,
 * including the weight matrices and bias vectors of each layer.
 *
 * @param W1 Weight matrix for the first layer.
 * @param b1 Bias vector for the first layer.
 * @param W2 Weight matrix for the second layer.
 * @param b2 Bias vector for the second layer.
 * @param filename Name of the file to save the parameters.
 */
void saveParameters(const Eigen::MatrixXf& W1, const Eigen::VectorXf& b1, const Eigen::MatrixXf& W2, const Eigen::VectorXf& b2, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write matrix dimensions
    int rows_W1 = W1.rows(), cols_W1 = W1.cols();
    int rows_W2 = W2.rows(), cols_W2 = W2.cols();
    int rows_b1 = b1.rows();
    int rows_b2 = b2.rows();

    file.write(reinterpret_cast<const char*>(&rows_W1), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols_W1), sizeof(int));
    file.write(reinterpret_cast<const char*>(&rows_W2), sizeof(int));
    file.write(reinterpret_cast<const char*>(&cols_W2), sizeof(int));
    file.write(reinterpret_cast<const char*>(&rows_b1), sizeof(int));
    file.write(reinterpret_cast<const char*>(&rows_b2), sizeof(int));

    // Write matrix data
    file.write(reinterpret_cast<const char*>(W1.data()), sizeof(float) * W1.size());
    file.write(reinterpret_cast<const char*>(W2.data()), sizeof(float) * W2.size());
    file.write(reinterpret_cast<const char*>(b1.data()), sizeof(float) * b1.size());
    file.write(reinterpret_cast<const char*>(b2.data()), sizeof(float) * b2.size());

    file.close();
}

/**
 * @brief Load neural network parameters from a file
 *
 * This function reads neural network parameters from a binary file,
 * including the weight matrices and bias vectors of each layer.
 *
 * @param filename Name of the file containing the parameters.
 * @return A tuple containing the loaded parameters:
 *         - Weight matrix for the first layer.
 *         - Bias vector for the first layer.
 *         - Weight matrix for the second layer.
 *         - Bias vector for the second layer.
 * If the file fails to open or there's an error during reading,
 * empty matrices and vectors are returned.
 */
std::tuple<Eigen::MatrixXf, Eigen::VectorXf, Eigen::MatrixXf, Eigen::VectorXf>loadParameters(const std::string& filename) {
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf b1, b2;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return std::make_tuple(W1, b1, W2, b2);
    }

    // Read matrix dimensions
    int rows_W1, cols_W1, rows_W2, cols_W2, rows_b1, rows_b2;
    file.read(reinterpret_cast<char*>(&rows_W1), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols_W1), sizeof(int));
    file.read(reinterpret_cast<char*>(&rows_W2), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols_W2), sizeof(int));
    file.read(reinterpret_cast<char*>(&rows_b1), sizeof(int));
    file.read(reinterpret_cast<char*>(&rows_b2), sizeof(int));

    // Resize matrices and vectors
    W1.resize(rows_W1, cols_W1);
    W2.resize(rows_W2, cols_W2);
    b1.resize(rows_b1);
    b2.resize(rows_b2);

    // Read matrix data
    file.read(reinterpret_cast<char*>(W1.data()), sizeof(float) * W1.size());
    file.read(reinterpret_cast<char*>(W2.data()), sizeof(float) * W2.size());
    file.read(reinterpret_cast<char*>(b1.data()), sizeof(float) * b1.size());
    file.read(reinterpret_cast<char*>(b2.data()), sizeof(float) * b2.size());

    file.close();
    return std::make_tuple(W1, b1, W2, b2);
}