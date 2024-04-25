#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>

#include "../include/dataset_utils.h"
#include "../include/helpers.h"


/* TODO:
 * - randomise order of images&labels
 */

// Temp func to print out image
void savePGM(const std::string& filename, const Eigen::MatrixXf& image) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write PGM header
    file << "P2" << std::endl; // PGM magic number
    file << "28 28" << std::endl; // Width and Height
    file << "255" << std::endl; // Maximum intensity value

    std::cout << image.rows() << std::endl;
    std::cout << image.cols() << std::endl;
    // Write pixel values (scaled by 255)
    for (int i = 0; i < image.rows(); ++i) {
        for (int j = 0; j < image.cols(); ++j) {
            file << image(i, j) * 255 << " ";
        }
        file << std::endl;
    }

    file.close();
}

int main() {

    std::string imageDataFile = "../data/train-images-idx3-ubyte";
    std::string labelDataFile = "../data/train-labels.idx1-ubyte";
    const int DATA_INDEX = 1960;

    // Load Data
    Eigen::MatrixXf mnistData = readData(imageDataFile);

    std::cout << mnistData.rows() << std::endl;
    std::cout << mnistData.cols() << std::endl;

    // Load label data
    std::vector<int> labels = readLabels(labelDataFile);
    int label = labels[DATA_INDEX];

    // mnistData is a matrix of pixel values
    Eigen::MatrixXf image = mnistData.col(DATA_INDEX);

    // Save the first item in the array as a PGM image
    std::string outputFilename = "first_image.pgm";
    savePGM(outputFilename, image);

    std::cout << "Label for image at index 10: " << label << std::endl;

    // Initialize parameters
    Eigen::MatrixXf W1;
    Eigen::MatrixXf b1;
    Eigen::MatrixXf W2;
    Eigen::MatrixXf b2;

    std::tie(W1, b1, W2, b2) = initParams();

    // Get the shape of W1
    int rows = W1.rows();
    int cols = W1.cols();

    // Print the shape of W1
    std::cout << "Shape of W1: " << rows << "x" << cols << std::endl;

    return 0;
}

