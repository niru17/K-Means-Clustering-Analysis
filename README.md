# K-Means-Clustering-Analysis
This Python script performs k-means clustering analysis on a dataset to identify the optimal number of clusters. It calculates the sum of squared errors (SSE) for different numbers of clusters and plots the results to determine the most suitable number of clusters.

Requirements:
- Python 3.x
- pandas
- numpy
- matplotlib

Usage

1. Data Preparation:

- Ensure the dataset file is available in the same directory as the script.
- The dataset should contain numerical data points with each row representing a data point.

2. Run the Script:
- Execute the Python script to perform k-means clustering analysis.
- Follow the prompts to enter the filename of the dataset.

The script will display the SSE for different numbers of clusters and plot the results.

3. Interpret Results:
- Analyze the plot to determine the optimal number of clusters based on the elbow method.
- The point where the plot shows diminishing returns in SSE reduction indicates the optimal number of clusters.

4. Features:
- File Loading: Load data from a text file containing numerical data points.
- Initialization: Initialize clusters with distinct data points as centroids.
- K-Means Iteration: Perform k-means iterations to assign data points to clusters and update centroids.
- SSE Calculation: Calculate the sum of squared errors (SSE) for different numbers of clusters.
- Visualization: Plot the SSE for various numbers of clusters to determine the optimal number.

5. Output:
- Training and Validation Accuracy: Display the accuracy scores of the SVM classifier on the training and validation sets.
- Confusion Matrices: Visualize the confusion matrices for the training, validation, and dummy test data.
- Cross-Validation Scores: Calculate and display the cross-validation scores to assess model robustness.
- Dummy Test Accuracy: Evaluate the accuracy of the SVM classifier on the provided dummy test data.
