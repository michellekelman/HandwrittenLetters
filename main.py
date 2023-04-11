import pandas as pd
import numpy as np
# from sklearn import _____
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    # loading data
    train = pd.read_csv('data/emnist-balanced-train.csv')
    test = pd.read_csv('data/emnist-balanced-test.csv')
    print("Number of examples in train:", len(train))
    print("Number of examples in test:", len(test))

    # train and test 2 models with our own algorithm implementation
    # # # confusion matrices
    # # # accuracy
    # # # TPR and FPR
    # # # area under ROC curvegit

    # train and test 2 models with scikit-learn mirroring our algorithms
    # # # confusion matrices
    # # # accuracy
    # # # TPR and FPR
    # # # area under ROC curve