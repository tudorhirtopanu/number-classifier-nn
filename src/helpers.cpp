#include "../include/helpers.h"
#include <Eigen/Core>
#include <random>
#include <iostream>

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
    Eigen::MatrixXf W1 = Eigen::MatrixXf::NullaryExpr(10, 784, [&](){return dis(gen);});
    Eigen::VectorXf b1 = Eigen::MatrixXf::NullaryExpr(10, 1,  [&](){return dis(gen);});
    Eigen::MatrixXf W2 = Eigen::MatrixXf::NullaryExpr(10, 10, [&](){return dis(gen);});
    Eigen::VectorXf b2 = Eigen::MatrixXf::NullaryExpr(10, 1, [&](){return dis(gen);});


    std::cout << "Initialized parameters:" << std::endl;
    std::cout << "W1:\n" << W1 << std::endl;
    std::cout << "b1:\n" << b1 << std::endl;
    std::cout << "W2:\n" << W2 << std::endl;
    std::cout << "b2:\n" << b2 << std::endl;


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

/**
 * @brief Get predictions from the output of a neural network.
 *
 * This function computes predictions from the output of a neural network by finding
 * the index of the maximum value in each column of the output matrix A2.
 *
 * The index of the maximum value also corresponds to the number(0-9) that the prediction is for,
 * if the highest score is at index 4, then it is predicting a number 4
 *
 * @param A2 The output matrix of shape (num_classes, num_samples) from the neural network.
 * @return A vector of predicted class labels, where each element represents the predicted class
 *         for the corresponding sample.
 */
Eigen::VectorXi getPredictions(const Eigen::MatrixXf& A2) {
    Eigen::VectorXi predictions(A2.cols());

    for (int i = 0; i < A2.cols(); ++i) {

        // Find the index of the maximum value in each column of A2
        float maxVal = A2.col(i).maxCoeff();
        for (int j = 0; j < A2.rows(); ++j) {
            if (A2(j, i) == maxVal) {
                predictions(i) = j;
                break;
            }
        }
    }
    return predictions;
}

/**
 * @brief Calculate accuracy of predictions.
 *
 * This function calculates the accuracy of predictions by comparing the predicted
 * values with the true labels and computing the proportion of correct predictions.
 *
 * @param predictions The vector of predicted class labels.
 * @param Y The vector of true class labels.
 * @return The accuracy of predictions, defined as the proportion of correct predictions.
 */
double getAccuracy(const Eigen::VectorXi& predictions, const Eigen::VectorXi& Y) {

    int numSamples = Y.size();
    int numCorrect = 0;

    for (int i = 0; i < numSamples; ++i) {
        if (predictions(i) == Y(i)) {
            numCorrect++;
        }
    }

    return static_cast<double>(numCorrect) / numSamples;
}
