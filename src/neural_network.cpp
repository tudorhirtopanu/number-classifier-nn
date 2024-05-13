
#include "../include/neural_network.h"
#include "../include/activation_functions.h"
#include "../include/helpers.h"
#include "../include/dataset_utils.h"

#include <Eigen/Core>
#include <iostream>



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
#include <chrono>

std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> forwardPropagation(const Eigen::MatrixXf& W1,
                                                                                                  const Eigen::MatrixXf& b1,
                                                                                                  const Eigen::MatrixXf& W2,
                                                                                                  const Eigen::MatrixXf& b2,
                                                                                                  const Eigen::MatrixXf& X){


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

/**
 * @brief Backward propagation for neural network
 *
 * This function performs backward propagation for the neural network,
 * computing the gradients of the cost function with respect to the
 * parameters (weights and biases) of the network.
 *
 * @param Z1 Activation values of the first hidden layer.
 * @param A1 Output values of the first hidden layer.
 * @param Z2 Activation values of the output layer.
 * @param A2 Output values of the output layer.
 * @param W1 Weight matrix for the first layer.
 * @param W2 Weight matrix for the second layer.
 * @param X Input data matrix.
 * @param Y Vector of true labels.
 *
 * @return A tuple containing the gradients of the cost function with respect to the parameters:
 *         - Gradient of the cost function with respect to the weights of the first layer.
 *         - Gradient of the cost function with respect to the biases of the first layer.
 *         - Gradient of the cost function with respect to the weights of the second layer.
 *         - Gradient of the cost function with respect to the biases of the second layer.
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> backwardPropagation(Eigen::MatrixXf Z1,Eigen::MatrixXf A1,Eigen::MatrixXf Z2,Eigen::MatrixXf A2,
                                                                                                   Eigen::MatrixXf W1,Eigen::MatrixXf W2,Eigen::MatrixXf X,Eigen::VectorXi Y){

    // Calculate the number of training examples
    float m = Y.size();

    // One hot encode labels
    Eigen::MatrixXi oneHotY = oneHotEncode(Y);

    Eigen::MatrixXf dZ2 = A2 - oneHotY.cast<float>();

    Eigen::MatrixXf dW2 = (1 / m) *dZ2 *A1.transpose();

    Eigen::VectorXf db2 = (1 / m) * dZ2.rowwise().sum();

    Eigen::MatrixXf dZ1 =  W2.transpose() * dZ2; // Dot product
    dZ1 = dZ1.array() * ReLU_derivative(Z1).array();    // Element-wise multiplication

    Eigen::MatrixXf dW1 = (1/m) * dZ1 * X.transpose();

    Eigen::VectorXf db1 = (1 / m) * dZ1.rowwise().sum();

    return std::make_tuple(dW1, db1, dW2, db2);
}

/**
 * @brief Update parameters for the neural network.
 *
 * This function updates the parameters (weights and biases) for the neural network based on the calculated gradients and a specified learning rate (alpha).
 * The parameters are updated using gradient descent, where each parameter is adjusted by subtracting the product of the gradient and the learning rate.
 *
 * @param W1 The weight matrix for the first layer
 * @param b1 The bias vector for the first layer
 * @param W2 The weight matrix for the second layer
 * @param b2 The bias vector for the second layer
 * @param dW1 The gradient of the loss with respect to W1
 * @param db1 The gradient of the loss with respect to b1
 * @param dW2 The gradient of the loss with respect to W2
 * @param db2 The gradient of the loss with respect to b2
 * @param alpha The learning rate for gradient descent
 *
 * @return A tuple containing the updated parameters:
 *         - W1: The updated weight matrix for the first layer
 *         - b1: The updated bias vector for the first layer
 *         - W2: The updated weight matrix for the second layer
 *         - b2: The updated bias vector for the second layer
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> updateParameters(Eigen::MatrixXf W1,Eigen::MatrixXf b1,Eigen::MatrixXf W2,
                                                                                                Eigen::MatrixXf b2,Eigen::MatrixXf dW1,Eigen::MatrixXf db1,
                                                                                                Eigen::MatrixXf dW2,Eigen::MatrixXf db2,float alpha){

    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;

    return std::make_tuple(W1, b1, W2, b2);
}

/**
 * @brief Perform gradient descent optimization for the neural network.
 *
 * This function optimizes the neural network parameters using gradient descent.
 * It updates the parameters (weights and biases) iteratively based on the gradients
 * of the cost function with respect to the parameters.
 *
 * @param X The input data matrix.
 * @param Y The vector of true class labels.
 * @param alpha The learning rate for gradient descent.
 * @param iterations The number of iterations for gradient descent.
 *
 * @return A tuple containing the optimized parameters:
 *         - W1: The optimized weight matrix for the first layer.
 *         - b1: The optimized bias vector for the first layer.
 *         - W2: The optimized weight matrix for the second layer.
 *         - b2: The optimized bias vector for the second layer.
 */
std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf> gradientDescent(Eigen::MatrixXf X,Eigen::VectorXi Y,Eigen::MatrixXf valX,Eigen::VectorXi valY, float alpha, int iterations){

    // initialise parameters
    Eigen::MatrixXf W1;
    Eigen::MatrixXf b1;
    Eigen::MatrixXf W2;
    Eigen::MatrixXf b2;

    std::tie(W1, b1, W2, b2) = initParams();

    for(int i = 0; i<iterations; i++){

        shuffleDataAndLabels(X, Y);
        shuffleDataAndLabels(valX, valY);

        Eigen::MatrixXf Z1; // pre activation value of neurons in first hidden layer
        Eigen::MatrixXf A1; // activated/output value of neurons in first hidden layer
        Eigen::MatrixXf Z2; // pre activation value of neurons in second hidden layer
        Eigen::MatrixXf A2; // activated/output value of neurons in second hidden layer

        std::tie(Z1, A1, Z2, A2) = forwardPropagation(W1, b1, W2, b2, X);

        Eigen::MatrixXf dW1; // gradient of the cost function with respect to the weights of the first layer.
        Eigen::MatrixXf db1; // gradient of the cost function with respect to the biases of the first layer.
        Eigen::MatrixXf dW2; // gradient of the cost function with respect to the weights of the second layer.
        Eigen::MatrixXf db2; // gradient of the cost function with respect to the biases of the second layer.

        std::tie(dW1, db1, dW2, db2) = backwardPropagation(Z1, A1, Z2, A2, W1, W2, X, Y);

        std::tie(W1, b1, W2, b2) = updateParameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);

        if((i+1)%10 == 0 || i == 0){
            // Calculate accuracy on validation set
            Eigen::MatrixXf valZ1; // pre activation value of neurons in first hidden layer
            Eigen::MatrixXf valA1; // activated/output value of neurons in first hidden layer
            Eigen::MatrixXf valZ2; // pre activation value of neurons in second hidden layer
            Eigen::MatrixXf valA2;
            std::tie(valZ1, valA1, valZ2, valA2) = forwardPropagation(W1, b1, W2, b2, valX);
            Eigen::VectorXi valPredictions = getPredictions(valA2);
            double valAccuracy = getAccuracy(valPredictions, valY);

            Eigen::VectorXi predictions = getPredictions(A2);
            double accuracy = getAccuracy(predictions, Y);
            std::cout << "Iteration: " << i+1 << ", Accuracy: " << accuracy << ", Validation Accuracy: " << valAccuracy << std::endl;
        }


    }

    return std::tie(W1, b1, W2, b2);
}

/**
 * @brief Run an image through the neural network and obtain the output.
 *
 * This function takes an image as input and passes it through the neural network
 * to obtain the output of the network.
 *
 * @param image The input image matrix.
 * @param W1 The weight matrix for the first layer.
 * @param b1 The bias vector for the first layer.
 * @param W2 The weight matrix for the second layer.
 * @param b2 The bias vector for the second layer.
 *
 * @return The output of the neural network after processing the input image.
 */

Eigen::MatrixXf runImageThroughNetwork(const Eigen::MatrixXf& image, const Eigen::MatrixXf& W1, const Eigen::MatrixXf& b1, const Eigen::MatrixXf& W2, const Eigen::MatrixXf& b2) {

    // Call the forwardPropagation function with the image data and network parameters
    return std::get<3>(forwardPropagation(W1, b1, W2, b2, image));
}

