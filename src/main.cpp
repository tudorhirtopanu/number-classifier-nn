#include <iostream>
#include <fstream>

#include "../include/dataset_utils.h"
#include "../include/helpers.h"
#include "../include/neural_network.h"
#include "../include/parameter_handler.h"

int main() {

    /**
     * Setup
     */

    // files for training images & labels
    std::string imageDataFile = "../data/train-images-idx3-ubyte";
    std::string labelDataFile = "../data/train-labels.idx1-ubyte";

    // files for testing images & labels
    std::string testImageDataFile = "../data/t10k-images-idx3-ubyte";
    std::string testLabelDataFile = "../data/t10k-labels.idx1-ubyte";

    // Load training images & labels
    Eigen::MatrixXf trainingData = readData(imageDataFile);
    Eigen::VectorXi labels = readLabels(labelDataFile);

    // Load test images & labels
    Eigen::MatrixXf testingData = readData(testImageDataFile);
    Eigen::VectorXi testingLabels = readLabels(testLabelDataFile);

    /**
     * Training Model
     *
     * -train the neural network and save parameters in the 'models' folder
     */


    Eigen::MatrixXf W1, b1, W2, b2;

    std::tie(W1, b1, W2, b2) = gradientDescent(trainingData, labels,testingData,testingLabels, 0.10, 250);

    saveParameters(W1, b1, W2, b2, "../models/params4.bin");



    /**
     * Testing Model
     *
     * -test the model by inputting an image from the testing data set
     */

/*
    const int TEST_DATA_INDEX = 103;
    Eigen::VectorXf testImage = testingData.col(TEST_DATA_INDEX);

    int testLabel = testingLabels(TEST_DATA_INDEX, 0);

    std::cout << "Label for image " << testLabel << std::endl;

    // Save the item as a PGM image (to provide visual of data item)
    savePGM("image.pgm", testImage);

    Eigen::MatrixXf W1, b1, W2, b2;
    std::tie(W1, b1, W2, b2) = loadParameters("../models/params3.bin");

    Eigen::MatrixXf result = runImageThroughNetwork(testImage,W1, b1, W2, b2);

    int maxIndex = findMaxIndex(result);

    std::cout << "Result:\n " << result << std::endl;
    std::cout << "\nPredicted Number: " << maxIndex << std::endl;

*/
    return 0;
}

