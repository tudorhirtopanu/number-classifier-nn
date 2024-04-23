#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "../include/dataset_utils.h"


int main() {

    std::string filename = "../data/train-images-idx3-ubyte";

    Eigen::MatrixXf mnistData = readData(filename);

    std::cout << "Loaded MNIST data from " << filename << std::endl;
    std::cout << "Number of images: " << mnistData.cols() << std::endl;
    std::cout << "Image size: " << mnistData.rows() << "x1 (flattened)" << std::endl;

    std::cout << "Pixel values of the first flattened image:" << std::endl;
    for (int i = 0; i < mnistData.rows(); ++i) {
        std::cout << mnistData(i, 4) << " ";
    }
    std::cout << std::endl;


    return 0;
}
