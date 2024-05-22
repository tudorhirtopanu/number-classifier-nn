# Number Classifier Neural Network

This repository contains a Neural Network written from scratch in C++ using the Eigen library. This project demonstrates a simple but effective approach to classifying MNIST handwritten digits.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
  - [CLion](#clion)
  - [VSCode](#vscode)
- [Usage](#usage)

## Introduction

This project demonstrates a neural network for digit classification. The neural network consists of an input layer (784 nodes), one hidden layer (10 nodes), and an output layer (10 nodes). It uses the ReLU activation function for the hidden layers and softmax for the output layer.

## Features

- Forward and backward propagation
- Gradient descent optimization
- Training and testing on MNIST dataset
- Validation set usage to detect overtraining
- Accuracy Calculation

## Dependencies

- [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- CMake (for VSCode)

## Installation

### CLion

To set up and run the project in CLion:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tudorhirtopanu/number-classifier-nn.git

2. **Open the project in CLion:**
- Open CLion.
- Select "Open" from the "Welcome to CLion" screen or from the "File" menu.
- Navigate to the cloned project directory and open it.

3. **Download and add Eigen:**
- Go to the Eigen main page and download the latest stable release.
- Modify the CMakeLists.txt to include the path to the downloaded Eigen folder:
  ```bash
  set(EIGEN_ROOT_DIR "/Path/to/eigen-3.4.0/")

4. **Choosing between training and testing:**
- In main.cpp, set the Mode to either TRAIN or TEST.
- If testing a model, set the directory in SAVED_MODEL.
- If training a new model, set NEW_MODEL_NAME.

5. **Click on the run icon to build and run the project:**

### VSCode

To set up and run the project in VSCode:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tudorhirtopanu/number-classifier-nn.git

2. **Open the project in VSCode**:
- Open VSCode.
- Select "Open" from the Start screen or from the "File" menu.
- Navigate to the cloned project directory and open it.

3. **Download and add Eigen**:
- Go to the Eigen main page and download the latest stable release.
- Modify the `CMakeLists.txt` to include the path to the downloaded Eigen folder:
  ```cmake
  set(EIGEN_ROOT_DIR "/Path/to/eigen-3.4.0/")

4. **Configuring and Building the Project**
- Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and select CMake: Configure.
- Open the Command Palette again and select CMake: Build.

***For Unix-like systems:***
```sh
   CMake .
   make
```
***For Windows:***
- Ensure you have a suitable compiler like MinGW or MSVC.
- The build should be handled by VSCode CMake Tools.

5. **Running the Project**
- Run the executable generated, e.g. ```./NumberClassiferNN``` or ```./NumberClassiferNN.exe```
- You may need to adjust data file paths in main.cpp to be relative to your CMake run directory

## Usage

1. **Set the Mode in `main.cpp`:**
   - For training, set `Mode` to `TRAIN` and specify `NEW_MODEL_NAME`.
   - For testing, set `Mode` to `TEST` and specify `SAVED_MODEL`.

2. **Set the `EPOCHS` and `LEARN_RATE` in `main.cpp`:**
3. **If testing, set `TEST_DATA_INDEX` in `main.cpp`, to choose a specific image to run through the neural network:**

4. **Run the project using your chosen IDE's build and run tools.**
