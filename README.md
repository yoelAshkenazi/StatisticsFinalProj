# Statistics Final Project
## Yoel Ashkenazi & Tamar Doron

## Project Description

Hiring Forecaster is a project that utilizes data and machine learning models to better understand which factors significantly impact hiring decisions. By analyzing various data points, this project aims to provide insights that can help streamline and improve the hiring process.

## Installation Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yoelAshkenazi/StatisticsFinalProj.git
    ```
2. Install the required dependencies:
    ```bash
    pip install numpy pandas matplotlib
    ```

## Usage Instructions

To explore the detailed documentation of every method and statistical analysis, as well as to execute the necessary code, refer to the `main.py` file. This file contains comprehensive information and instructions on how to run the various components of the project.

To see outputs, you need to call one of the methods in `main.py`:

1. `make_pca`: Plots a PCA representation of the data.
2. `check_feature_dependence`: Uses the chi-square test to determine what features depend on each other.
3. `check_feature_distribution`: Uses the Kolmogorov-Smirnov test to determine if each feature is normally or uniformly distributed.
4. `check_feature_improvement`: Implements a pipeline to check if a specified feature improves accuracy with statistical significance.
5. `test_anova`: Compares results for several features to determine if they equally contributed.
6. `check_biases`: Plots more analysis about the percentage of acceptance among each feature.

To run any of these methods, simply execute `main.py` and call the desired method.
We recommend cloning the git and opening it in an IDE, as there is a detailed documentation of everything in the 'main.py' file, including arguments for each method.
