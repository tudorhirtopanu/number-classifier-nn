#include <iostream>
#include <sstream>

using namespace std;

/*
 * Splits a string by delimiter and returns tokens as vector
 * @param s String to split
 * @param delimiter The character delimiter
 * @return A vector containing the split tokens
 */
vector<int> split(const string &s, char delimiter){

    const int numTokens = 5;
    // Define a vector of integers to store the split parts
    vector<int> tokens;
    tokens.reserve(numTokens);

    // Define a string variable to store each split part
    string token;

    // Create an input string stream to parse original string
    istringstream token_stream(s);

    // Continue reading from the input string stream until there are no more delimiters
    while(getline(token_stream, token, delimiter)){
        // Convert each token from string to integer and add it to tokens vector
        tokens.push_back(stoi(token));
    }

    // Return vector containing split parts
    return tokens;
}

/*
 * Transposes a 2D vector, making its rows the columns
 * @param input 2D vector
 * @return input 2D vector transposed
 */
std::vector<std::vector<int>> transpose(const std::vector<std::vector<int>> &input){

    // Get dimensions of input vector
    // Use unsigned type to ensure compatibility with vector sizes
    std::vector<std::vector<int>>::size_type rows = input.size();
    std::vector<std::vector<int>>::size_type columns = input[0].size();

    // Create a new vector to store the transposed data
    std::vector<std::vector<int>> transposed(columns, std::vector<int>(rows));

    // Iterate over input vector and populate the transposed vector
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<columns; j++){
            transposed[j][i] = input[i][j];
        }
    }

    return transposed;
}

/*
 * TODO:
 * - Load line by line and create 1D vector of integers
 * - Determine number of rows and columns needed for the 2D array
 * - Allocate memory for 2d array with correct dimensions
 * - Iterate through 1D arrays and assign their elements to corresponding column in the 2D array
 */

int main() {

    string numbers = "1,4,5,7,9";
    vector<int> number_vector = split(numbers, ',');
    //cout << number_vector[1] + number_vector[2];

    // Create a 2D vector with 2 rows and 4 columns
    vector<std::vector<int>> myVector = {
            {4, 4, 4, 4},   // First row with 1s
            {8, 8, 8, 8}    // Second row with 2s
    };

    std::vector<std::vector<int>> transposedVector = transpose(myVector);

    // Print the 2D vector
    for (const auto& row : transposedVector) {
        for (int value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}
