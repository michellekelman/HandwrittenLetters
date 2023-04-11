import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    # loading data
    ytrn = np.load('numpy/ytrn.npy')
    Xtrn = np.load('numpy/Xtrn.npy')
    ytst = np.load('numpy/ytst.npy')
    Xtst = np.load('numpy/Xtst.npy')
    print("Data loaded successfully")

    # train and test 2 models with our own algorithm implementation
    # # # confusion matrices
    # # # accuracy
    # # # TPR and FPR
    # # # area under ROC curvegit

    # train and test 2 models with scikit-learn mirroring our algorithms
    relu = MLPClassifier().fit(Xtrn, ytrn)
    print("Train successful")
    ypred = relu.predict(Xtst)
    print("Predict successful")
    accuracy = relu.score(Xtst, ytst)
    print(accuracy)
    # # # confusion matrices
    # # # accuracy
    # # # TPR and FPR
    # # # area under ROC curve