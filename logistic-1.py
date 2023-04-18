# TRAIN AND TEST 2 MODELS WITH OUR NEURAL NETWORK IMPLEMENTATION
# 1 Hidden Layer Neural Network implementation
# Sources:
# # http://neuralnetworksanddeeplearning.com/chap1.html#:~:text=The%20idea%20is%20to%20take,rules%20for%20recognizing%20handwritten%20digits.
# # https://www.kaggle.com/code/sanwal092/3-layer-neural-network-from-scratch/notebook
# # https://towardsdatascience.com/building-a-neural-network-with-a-single-hidden-layer-using-numpy-923be1180dbf

import numpy as np
import random
np.random.seed(0)
import warnings
warnings.filterwarnings('ignore')

class NeuralNetwork():
    def __init__(self, inputNum, hiddenNum, outputNum):
        self.layers = 3
        self.b1 = np.zeros((hiddenNum, 1))
        self.w1 = np.random.randn(hiddenNum, inputNum)*0.01
        self.b2 = np.zeros((outputNum, 1))
        self.w2 = np.random.randn(outputNum, hiddenNum)*0.01

    def sigmoid(self, x):
        sigmoid = 1.0 / (1.0+np.exp(-x))
        return sigmoid
    
    def forward(self, X):
        z1 = np.matmul(self.w1, X.T) + self.b1
        self.z1 = z1
        a1 = self.sigmoid(z1)
        z2 = np.matmul(self.w2, a1) + self.b2
        a2 = self.sigmoid(z2)
        return a1, a2
    
    def compute_loss(self, y, a2):
        loss_xi = np.multiply(y, np.log(a2)) + np.multiply((1.0-y), np.log(1.0-a2))
        n = y.shape[1]
        loss = -(1.0/n) * np.sum(loss_xi)
        return loss
    
    def backward(self, X, y, a1, a2):
        error = a2-y
        # delta2 = error * a2 * (1.0-a2)
        delta2 = error
        delta1 = np.matmul(self.w2.T, delta2) * a1 * (1.0-a1)
        n = y.shape[1]
        dw2 = (1.0/n) * np.matmul(delta2, a1.T)
        db2 = (1.0/n) * np.sum(delta2, axis=1, keepdims=True)
        dw1 = (1.0/n) * np.matmul(delta1, X)
        db1 = (1.0/n) * np.sum(delta1, axis=1, keepdims=True)
        return db1, dw1, db2, dw2
    
    def gradient(self, db1, dw1, db2, dw2, lr=0.01):
        self.b1 = self.b1 - (lr * db1)
        self.w1 = self.w1 - (lr * dw1)
        self.b2 = self.b2 - (lr * db2)
        self.w2 = self.w2 - (lr * dw2)
    
    def fit(self, X, y, epochs=1000, lr=0.01):
        for i in range(epochs):
            a1, a2 = self.forward(X)
            loss = self.compute_loss(y, a2)
            db1, dw1, db2, dw2 = self.backward(X, y, a1, a2)
            self.gradient(db1, dw1, db2, dw2, lr)
            if i % 10 == 0:
                print("Epoch", i, "loss:", loss)

    def predict(self, X):
        a1, a2 = self.forward(X)
        # y_pred = a2
        y_pred = np.argmax(a2, axis=0)
        return y_pred

    def compute_error(self, y_true, y_pred):
        error = (1.0/len(y_true)) * sum(y_true!=y_pred)
        return error

if __name__ == '__main__':
    # nupmy option for debugging
    # np.set_printoptions(threshold=np.inf)

    # number of classes
    numClasses = 10

    # loading data
    ytrn0 = np.load('numpy/ytrn.npy')
    Xtrn = np.load('numpy/Xtrn.npy')
    ytst0 = np.load('numpy/ytst.npy')
    Xtst = np.load('numpy/Xtst.npy')
    print("Data loaded successfully")

    # one hot encoding for y train data
    ytrn1 = ytrn0.reshape(1, ytrn0.shape[0])
    ytrn = np.eye(numClasses)[ytrn1]
    ytrn = ytrn.T.reshape(numClasses, ytrn0.shape[0])

    # one hot encoding for y test data
    ytst1 = ytst0.reshape(1, ytst0.shape[0])
    ytst = np.eye(numClasses)[ytst1]
    ytst = ytst.T.reshape(numClasses, ytst0.shape[0])

    # input num: number of features in the dataset
    inputNum = 28*28
    # hidden num: number of nodes in hidden layer
    hiddenNum = 300
    # output num: number of classes
    outputNum = numClasses

    # model
    model = NeuralNetwork(inputNum, hiddenNum, outputNum)
    model.fit(Xtrn, ytrn, epochs=1000, lr=0.1)

    # predictions
    y_pred_trn = model.predict(Xtrn)
    y_pred_tst = model.predict(Xtst)

    # scoring
    error_trn = model.compute_error(ytrn0, y_pred_trn)
    error_tst = model.compute_error(ytst0, y_pred_tst)
    print("Train Error: ", error_trn)
    print("Test Error: ", error_tst)
    
    # # # confusion matrices
    # # # accuracy
    # # # TPR and FPR
    # # # area under ROC curve