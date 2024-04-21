#include "../include/dataset_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

/*
 * TODO:
 * - Get number of data items
 * - create a vector of indices
 * - shuffle indices
 * - use that to load batches
 */

std::vector<int> generateRandomIndices(int itemCount){

    // Create vector of indices from 0 to itemCount-1
    std::vector<int> indices(itemCount);
    iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    return indices;
}

