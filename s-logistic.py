# TRAIN AND TEST 2 MODELS WITH SCIKIT_LEARN MIRRORING OUR NEURAL NETWORK IMPLEMENTATION

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

    # model
    # relu = MLPClassifier().fit(Xtrn, ytrn)
    relu = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000).fit(Xtrn, ytrn)
    print("Train successful")
    
    # predictions
    ypred = relu.predict(Xtst)
    print("Predict successful")

    # scoring
    accuracy = relu.score(Xtst, ytst)
    print(accuracy)

    # confusion matrices
    # accuracy
    # TPR and FPR
    # area under ROC curve