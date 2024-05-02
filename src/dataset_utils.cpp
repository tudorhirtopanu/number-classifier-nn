#include "../include/dataset_utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>

/**
 * @brief Convert endianness of a 32-bit integer.
 *
 * This function converts the endianness of a 32-bit integer from big-endian to little-endian or vice versa.
 *
 * @param value The 32-bit integer whose endianness is to be converted.
 * @return The 32-bit integer with its endianness converted.
 */
int32_t convertEndian(int32_t value) {
    return ((value >> 24) & 0xff) | ((value << 8) & 0xff0000) |
           ((value >> 8) & 0xff00) | ((value << 24) & 0xff000000);
}

/**
 * @brief Read the header of a dataset file.
 *
 * This function reads the header of a dataset file
 *
 * @param file An input file stream connected to the dataset file.
 * @return A tuple containing the magic number, number of images, number of rows, and number of columns.
 */
std::tuple<int, int, int, int> readDatasetHeader(std::ifstream& file) {
    int magicNumber, numImages, numRows, numCols;

    // Read the magic number, number of images, number of rows, and number of columns
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

    // Convert from big-endian to little-endian
    magicNumber = convertEndian(magicNumber);
    numImages = convertEndian(numImages);
    numRows = convertEndian(numRows);
    numCols = convertEndian(numCols);

    return std::make_tuple(magicNumber, numImages, numRows, numCols);
}

/**
 * @brief Read image data from an IDX file into an Eigen matrix.
 *
 * This function reads mnist image data from an IDX file, normalises the pixel values
 * and stores them in an Eigen matrix.
 *
 * @param filename The name of the IDX file to read.
 * @return An Eigen matrix containing the image data.
 */
Eigen::MatrixXf readData(const std::string& filename) {

    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    // Read the header of the dataset file
    int magicNumber, numImages, numRows, numCols;
    std::tie(magicNumber, numImages, numRows, numCols) = readDatasetHeader(file);

    // Check if the magic number indicates an IDX3-ubyte file
    if (magicNumber != 2051) {
        std::cerr << "Invalid magic number. This might not be an IDX3-ubyte file." << std::endl;
        exit(1);
    }

    // Initialize an Eigen matrix to store the image data
    Eigen::MatrixXf data(numRows * numCols, numImages);

    // Read image data pixel by pixel
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < numRows * numCols; ++j) {
            unsigned char pixel;
            // Read a single pixel value from the file
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            // Normalize pixel values and store it
            data(j, i) = static_cast<float>(pixel)/255.0f;
        }
    }

    file.close();
    return data;
}

/**
 * @brief Read label data from an IDX file into an eigen matrix.
 *
 * This function reads label data from an IDX file and stores it in an eigen matrix.
 *
 * @param filename The name of the IDX file to read.
 * @return A matrix containing the label data.
 */
Eigen::VectorXi readLabels(const std::string& filename) {
    Eigen::VectorXi labels;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return labels;
    }

    // Read the magic number
    int magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number); // Convert from big-endian to little-endian
    if (magic_number != 2049) {
        std::cerr << "Invalid magic number" << std::endl;
        return labels;
    }
    // Read the number of labels
    int num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels); // Convert from big-endian to little-endian

    labels.resize(num_labels);

    // Read the labels
    for (int i = 0; i < num_labels; ++i) {
        uint8_t label_byte;
        file.read(reinterpret_cast<char*>(&label_byte), sizeof(label_byte));
        labels(i, 0) = static_cast<int>(label_byte);
    }


    return labels;
}
