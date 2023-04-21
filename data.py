# PREPROCESS AND SAVE THE DATA

import numpy as np

if __name__ == '__main__':
    # loading data
    train = np.genfromtxt('data/emnist-balanced-train.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    test = np.genfromtxt('data/emnist-balanced-test.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    print("Number of examples in train:", len(train))
    print("Number of examples in test:", len(test))

    # splitting data
    ytrn = train[:, 0]
    Xtrn = train[:, 1:]
    ytst = test[:, 0]
    Xtst = test[:, 1:]
    print("Number of examples in y train:", len(ytrn))
    print("Number of examples in X train:", len(Xtrn))
    print("Number of examples in y test:", len(ytst))
    print("Number of examples in X test:", len(Xtst))

    # process X data to binary features
    itrn = np.where(Xtrn!=0)
    itst = np.where(Xtst!=0)
    Xtrn[itrn] = 1
    Xtst[itst] = 1

    # saving data as np arrays for future use
    np.save('numpy/ytrn.npy', ytrn)
    np.save('numpy/Xtrn.npy', Xtrn)
    np.save('numpy/ytst.npy', ytst)
    np.save('numpy/Xtst.npy', Xtst)