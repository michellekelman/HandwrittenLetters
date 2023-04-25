# 3 Hidden Layer Neural Network implementation with ReLU Activation Function
# Sources:
# # http://neuralnetworksanddeeplearning.com/chap1.html#:~:text=The%20idea%20is%20to%20take,rules%20for%20recognizing%20handwritten%20digits.
# # https://www.kaggle.com/code/sanwal092/3-layer-neural-network-from-scratch/notebook
# # https://towardsdatascience.com/building-a-neural-network-with-a-single-hidden-layer-using-numpy-923be1180dbf

import numpy as np
import random
np.random.seed(0)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class NeuralNetwork():
    def __init__(self, inputNum, hiddenNum1, hiddenNum2, hiddenNum3, outputNum):
        self.layers = 5
        self.b1 = np.zeros((hiddenNum1, 1))
        self.w1 = np.random.randn(hiddenNum1, inputNum)*0.01
        self.b2 = np.zeros((hiddenNum2, 1))
        self.w2 = np.random.randn(hiddenNum2, hiddenNum1)*0.01
        self.b3 = np.zeros((hiddenNum3, 1))
        self.w3 = np.random.randn(hiddenNum3, hiddenNum2)*0.01
        self.b4 = np.zeros((outputNum, 1))
        self.w4 = np.random.randn(outputNum, hiddenNum3)*0.01

    def relu(self, x):
        relu = np.maximum(np.zeros((x.shape[0], x.shape[1])), x)
        return relu
    
    def forward(self, X):
        z1 = np.matmul(self.w1, X.T) + self.b1
        a1 = self.relu(z1)
        z2 = np.matmul(self.w2, a1) + self.b2
        a2 = self.relu(z2)
        z3 = np.matmul(self.w3, a2) + self.b3
        a3 = self.relu(z3)
        z4 = np.matmul(self.w4, a3) + self.b4
        a4 = self.relu(z4)
        return a1, a2, a3, a4
    
    def compute_loss(self, y, a4):
        loss_xi = np.square(y-a4)
        n = y.shape[1]
        loss = -(1.0/n) * np.sum(loss_xi)
        return loss
    
    def backward(self, X, y, a1, a2, a3, a4):
        error = a4-y
        delta4 = error
        delta3 = np.matmul(self.w4.T, delta4) * a3 * (1.0-a3)
        delta2 = np.matmul(self.w3.T, delta3) * a2 * (1.0-a2)
        delta1 = np.matmul(self.w2.T, delta2) * a1 * (1.0-a1)
        n = y.shape[1]
        dw4 = (1.0/n) * np.matmul(delta4, a3.T)
        db4 = (1.0/n) * np.sum(delta4, axis=1, keepdims=True)
        dw3 = (1.0/n) * np.matmul(delta3, a2.T)
        db3 = (1.0/n) * np.sum(delta3, axis=1, keepdims=True)
        dw2 = (1.0/n) * np.matmul(delta2, a1.T)
        db2 = (1.0/n) * np.sum(delta2, axis=1, keepdims=True)
        dw1 = (1.0/n) * np.matmul(delta1, X)
        db1 = (1.0/n) * np.sum(delta1, axis=1, keepdims=True)
        return db1, dw1, db2, dw2, db3, dw3, db4, dw4
    
    def gradient(self, db1, dw1, db2, dw2, db3, dw3, db4, dw4, lr):
        self.b1 = self.b1 - (lr * db1)
        self.w1 = self.w1 - (lr * dw1)
        self.b2 = self.b2 - (lr * db2)
        self.w2 = self.w2 - (lr * dw2)
        self.b3 = self.b3 - (lr * db3)
        self.w3 = self.w3 - (lr * dw3)
        self.b4 = self.b4 - (lr * db4)
        self.w4 = self.w4 - (lr * dw4)
    
    def fit(self, X, y, epochs=1000, lr=0.01):
        for i in range(epochs):
            a1, a2, a3, a4 = self.forward(X)
            loss = self.compute_loss(y, a4)
            db1, dw1, db2, dw2, db3, dw3, db4, dw4 = self.backward(X, y, a1, a2, a3, a4)
            self.gradient(db1, dw1, db2, dw2, db3, dw3, db4, dw4, lr)
            if i % 10 == 0:
                print("Epoch", i, "loss:", loss)

    def predict(self, X):
        a1, a2, a3, a4 = self.forward(X)
        y_pred = np.argmax(a4, axis=0)
        return y_pred

    def confusion_matrix(self, y_true, y_pred, numClasses):
        matrix = np.zeros((numClasses, numClasses))
        for i in range(len(y_true)):
            matrix[y_true[i], y_pred[i]] += 1
        return matrix

    def compute_error(self, y_true, y_pred):
        error = (1.0/len(y_true)) * sum(y_true!=y_pred)
        return error
    
    def evaluation(self, matrix, numClasses, numPoints):
        tpr = np.zeros(numClasses)
        fpr = np.zeros(numClasses)
        precision = np.zeros(numClasses)
        recall = np.zeros(numClasses)
        f1 = np.zeros(numClasses)
        for i in range(numClasses):
            tp = matrix[i, i]
            fp = sum(matrix[j, i] for j in range(numClasses)) - tp
            fn = sum(matrix[i, j] for j in range(numClasses)) - tp
            tn = numPoints - tp - fp - fn
            tpr[i] = tp / (tp+fn)
            fpr[i] = fp / (fp+tn)
            precision[i] = tp / (tp+fp)
            recall[i] = tp / (tp+fn)
            f1[i] = 2*precision[i]*recall[i] / (precision[i]+recall[i])
        return tpr, fpr, precision, recall, f1
    
    def ROC(self, X, y_true, numClasses):
        a1, a2, a3, a4 = self.forward(X)
        y_pred = np.argmax(a4, axis=0)
        y_dist = np.max(a4, axis=0)
        ind = np.lexsort((-y_dist, y_pred))
        j = 0
        roc_areas = []
        for i in range(numClasses):
            tp = 0
            fp = 0
            fpt = 0
            areas = np.array([0])
            # tps = np.array([0])
            # fps = np.array([0])
            while j < len(y_true) and y_pred[ind[j]]==i:
                while j < len(y_true) and y_pred[ind[j]]==i and y_pred[ind[j]]==y_true[ind[j]]:
                    tp += 1
                    j += 1
                # tps = np.append(tps, tp)
                # fps = np.append(fps, fpt)
                while j < len(y_true) and y_pred[ind[j]]==i and y_pred[ind[j]]!=y_true[ind[j]]:
                    fp += 1
                    fpt += 1
                    j += 1
                # tps = np.append(tps, tp)
                # fps = np.append(fps, fpt)
                areas = np.append(areas, tp*fp)
                fp = 0
            areas = areas / (tp*fpt)
            roc_areas = np.append(roc_areas, sum(areas))
            # tps = tps / tp
            # fps = fps / fpt
            # plt.plot(fps, tps)
            # plt.show()
        return roc_areas

if __name__ == '__main__':
    # nupmy option for debugging
    # np.set_printoptions(threshold=np.inf)

    # number of classes
    numClasses = 47

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
    # hidden num: number of nodes in hidden layers
    hiddenNum1 = 150
    hiddenNum2 = 100
    hiddenNum3 = 50
    # output num: number of classes
    outputNum = numClasses

    # model
    model = NeuralNetwork(inputNum, hiddenNum1, hiddenNum2, hiddenNum3, outputNum)
    model.fit(Xtrn, ytrn, epochs=1000, lr=0.1)

    # predictions
    y_pred_trn = model.predict(Xtrn)
    y_pred_tst = model.predict(Xtst)

    # scoring
    # # # confusion matrices
    cfm = model.confusion_matrix(ytst0, y_pred_tst, numClasses)
    print("Matrix: ", cfm)
    ax = sns.heatmap(cfm, annot=True, fmt="g", cmap="flare")
    ax.set_title('3 Hidden Layers - ReLU Activation')
    ax.set(xlabel="Predicted Class", ylabel="True Class")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # # # accuracy
    error_trn = model.compute_error(ytrn0, y_pred_trn)
    error_tst = model.compute_error(ytst0, y_pred_tst)
    print("Train Accuracy: ", 1-error_trn)
    print("Test Accuracy: ", 1-error_tst)
    
    # # # TPR and FPR
    tpr, fpr, precision, recall, f1 = model.evaluation(cfm, numClasses, len(ytst0))
    print("TPR: ", tpr)
    print("Average TPR: ", sum(tpr)/numClasses)
    print("FPR: ", fpr)
    print("Average FPR: ", sum(fpr)/numClasses)
    print("Precision: ", precision)
    print("Average Precision: ", sum(precision)/numClasses)
    print("Recall: ", recall)
    print("Average Recall: ", sum(recall)/numClasses)
    print("F1: ", f1)
    print("Average F1: ", sum(f1)/numClasses)

    # # # area under ROC curve
    roc = model.ROC(Xtst, ytst0, numClasses)
    print("ROC: ", roc)
    print("Average ROC: ", sum(roc)/numClasses)

    plt.show()