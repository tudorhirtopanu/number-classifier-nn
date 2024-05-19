# Number Classifier Nueural Network

This repository contains a Neural Network written from scratch in C++ using the Eigen library. This project demonstrates a simple but effective approach to classifying MNIST handwritten digits.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation (CLion)](#installation-clion)

## Introduction

This project demonstrates a neural network for digit classification. The neural network consists of an input layer (784 nodes), one hidden layer (10 nodes), and an output layer (10 nodes). It uses the ReLU activation function for the hidden layers and softmax for the output layer.

## Features

- **Forward and backward propagation**
- **Gradient descent optimization**
- **Training and testing on MNIST dataset**
- **Accuracy Calculation**

## Dependencies

- **Eigen library**

## Installation (CLion)

To set up and run the project in CLion:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/tudorhirtopanu/number-classifier-nn.git
    ```

2. **Open the project in CLion**:
    - Open CLion.
    - Select "Open" from the "Welcome to CLion" screen or from the "File" menu.
    - Navigate to the cloned project directory and open it.

3. **Download the Eigen library**
    - Go to the Eigen main page and download the latest stable release
   ```sh
   https://eigen.tuxfamily.org/index.php?title=Main_Page
    ```
   
3. **Add Eigen to CMake.txt**:
    - Modify this line in CMake.txt to contain the path to inside the downloaded folder (You may need to reload CMake project afterwards)
   ```sh
   set(EIGEN_ROOT_DIR "/Path/to/eigen-3.4.0/")
    ```

4. **Configuring and Executing the Project**:

    #### In main.cpp:
    - Set the Mode to either TRAIN or TEST
    - If testing a model, set the directory in SAVED_MODEL
    - If training a new model, set NEW_MODEL_NAME
      
    #### Click on the run icon to build and run the project.



