# 1 Hidden Layer Scikit-Learn Neural Network with Logistic Activation Function

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def evaluation(matrix, numClasses, numPoints):
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

if __name__ == '__main__':
    # number of classes
    numClasses = 10

    # loading data
    ytrn = np.load('numpy/ytrn.npy')
    Xtrn = np.load('numpy/Xtrn.npy')
    ytst = np.load('numpy/ytst.npy')
    Xtst = np.load('numpy/Xtst.npy')
    print("Data loaded successfully")

    # model
    # relu = MLPClassifier().fit(Xtrn, ytrn)
    logistic = MLPClassifier(hidden_layer_sizes=(300,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000).fit(Xtrn, ytrn)
    print("Train successful")
    
    # predictions
    ypred = logistic.predict(Xtst)
    ydist = logistic.predict_proba(Xtst)
    print("Predict successful")

    # scoring
    # # # confusion matrices
    cfm = confusion_matrix(ytst, ypred)
    print("Matrix: ", cfm)
    ax = sns.heatmap(cfm, annot=True, cmap="flare")
    ax.set_title('Scikit-Learn Neural Network - Logistic Activation')
    ax.set(xlabel="Predicted Class", ylabel="True Class")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # # # accuracy
    accuracy = logistic.score(Xtst, ytst)
    print("Test Accuracy: ", accuracy)
    
    # # # TPR and FPR
    tpr, fpr, precision, recall, f1 = evaluation(cfm, numClasses, len(ytst))
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
    roc = roc_auc_score(ytst, ydist, average=None, multi_class='ovr')
    print("ROC: ", roc)
    print("Average ROC: ", sum(roc)/numClasses)

    plt.show()