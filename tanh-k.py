# Keras Convolutional Neural Network with Hyperbolic Tangent Activation Function
# Sources:
# # https://data-flair.training/blogs/handwritten-character-recognition-neural-network/

#import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

# loading data
ytrn = np.load('numpy/ytrn.npy')
Xtrn = np.load('numpy/Xtrn.npy')
ytst = np.load('numpy/ytst.npy')
Xtst = np.load('numpy/Xtst.npy')
print("Data loaded successfully")
Xtrn = np.reshape(Xtrn, (Xtrn.shape[0], 28,28))
Xtst = np.reshape(Xtst, (Xtst.shape[0], 28,28))
print("Train data shape: ", Xtrn.shape)
print("Test data shape: ", Xtst.shape)

# reshaping data
Xtrn = Xtrn.reshape(Xtrn.shape[0],Xtrn.shape[1],Xtrn.shape[2],1)
print("New shape of train data: ", Xtrn.shape)

Xtst = Xtst.reshape(Xtst.shape[0], Xtst.shape[1], Xtst.shape[2],1)
print("New shape of train data: ", Xtst.shape)

# number of classes
numClasses = 10

# categorizing labels
ytrnOHE = to_categorical(ytrn, num_classes = numClasses, dtype='int')
print("New shape of train labels: ", ytrnOHE.shape)

ytstOHE = to_categorical(ytst, num_classes = numClasses, dtype='int')
print("New shape of test labels: ", ytstOHE.shape)

# model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='tanh', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='tanh', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="tanh"))
model.add(Dense(128,activation ="tanh"))
model.add(Dense(numClasses,activation ="softmax"))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# predictions
history = model.fit(Xtrn, ytrnOHE, epochs=1, callbacks=[reduce_lr, early_stop], validation_data=(Xtst, ytstOHE))
ydist = model.predict(Xtst)
ypred = np.argmax(ydist, axis=1)
model.summary()

# scoring
# # # confusion matrices
cfm = confusion_matrix(ytst, ypred)
print("Matrix: ", cfm)
plt.figure(figsize=(10,8))
ax = sns.heatmap(cfm, annot=True, fmt="g", cmap="flare")
ax.set_title('Keras CNN - Hyperbolic Tangent Activation')
ax.set(xlabel="Predicted Class", ylabel="True Class")
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# # # accuracy
print("The training accuracy is :", history.history['accuracy'])
print("The testing accuracy is :", history.history['val_accuracy'])
print("The training loss is :", history.history['loss'])
print("The testing loss is :", history.history['val_loss'])

# # # TPR and FPR
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
lb = LabelBinarizer()
lb.fit(ytst)
ytst_lb = lb.transform(ytst)
ypred_lb = lb.transform(ypred)
roc = roc_auc_score(ytst_lb, ypred_lb, average=None, multi_class='ovr')
print("ROC: ", roc)
print("Average ROC: ", sum(roc)/numClasses)

plt.show()