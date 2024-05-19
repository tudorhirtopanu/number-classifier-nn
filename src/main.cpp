#include <iostream>
#include <fstream>

#include "../include/dataset_utils.h"
#include "../include/helpers.h"
#include "../include/neural_network.h"
#include "../include/parameter_handler.h"

enum Mode {
    TRAIN,
    TEST
};

int main() {

    /**
     * Setup
     */

    // Choose the model to load
    const std::string SAVED_MODEL = "../models/model.bin";

    // Set name for new models to save
    const std::string NEW_MODEL_NAME = "model";

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

    // Set the mode (TRAIN or TEST)
    Mode mode = Mode::TEST;
    /**
     * Training Model
     *
     * -train the neural network and save parameters in the 'models' folder
     */

    if (mode == Mode::TRAIN) {
        Eigen::MatrixXf W1, b1, W2, b2;
        std::tie(W1, b1, W2, b2) = gradientDescent(trainingData, labels, testingData, testingLabels, 0.15, 650);
        saveParameters(W1, b1, W2, b2, "../models/"+NEW_MODEL_NAME);
    }

    /**
     * Testing Model
     *
     * -test the model by inputting an image from the testing data set
     */
    if (mode == Mode::TEST) {
        const int TEST_DATA_INDEX = 113;
        Eigen::VectorXf testImage = testingData.col(TEST_DATA_INDEX);

        int testLabel = testingLabels(TEST_DATA_INDEX, 0);

        std::cout << "Label for image " << testLabel << std::endl;

        // Save the item as a PGM image (to provide visual of data item)
        savePGM("image.pgm", testImage);

        Eigen::MatrixXf W1, b1, W2, b2;
        std::tie(W1, b1, W2, b2) = loadParameters(SAVED_MODEL);

        Eigen::VectorXf result = runImageThroughNetwork(testImage, W1, b1, W2, b2);

        // Print the confidence scores for each digit
        std::cout << "Result:" << std::endl;
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                std::cout << "Confidence score for " << i << ": " << result(i, j) << std::endl;
            }
        }

        int maxIndex = findMaxIndex(result);
        std::cout << "\nPredicted Number: " << maxIndex << std::endl;
    }

    return 0;
}

