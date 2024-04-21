#include <iostream>
#include <vector>
#include "../include/dataset_utils.h"

int main() {

    // Define the number of data items
    const int ITEM_COUNT = 10;

    // Generate randomized indices
    std::vector<int> indices = generateRandomIndices(ITEM_COUNT);

    // Print the randomized indices
    std::cout << "Randomized indices:" << std::endl;
    for (int idx : indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    return 0;
}

