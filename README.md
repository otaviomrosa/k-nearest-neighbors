# k-Nearest Neighbors (k-NN) from Scratch

This project is a "from-scratch" implementation of the k-Nearest Neighbors (k-NN) algorithm, built using only **PyTorch** and **NumPy**.

## Project Goal

The goal was to build a k-NN classifier from its core principles to understand how it works at a mathematical level. The model is capable of:
- Loading a dataset (e.g., CIFAR-10).
- Calculating the Euclidean distance between all test and train samples.
- Finding the 'k' closest neighbors for each test sample.
- Predicting a class based on a majority vote from its neighbors.

## How It Works

The `knn.py` script leverages PyTorch's tensor operations for efficient computation:

1.  **Load Data:** Loads the CIFAR-10 dataset.
2.  **Calculate Distances:** Computes the L2 (Euclidean) distance between each test point and every training point. This is done with efficient tensor broadcasting.
3.  **Find Neighbors:** Uses `torch.topk` to find the indices of the 'k' nearest neighbors for each test point.
4.  **Vote for Class:** Gathers the labels of those neighbors and determines the most frequent class (the "vote").
5.  **Report Accuracy:** Compares the predicted labels to the true labels and prints the final accuracy.

## Results

The model achieved an accuracy of [XX]% on the CIFAR-10 test set.
