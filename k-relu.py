# TRAIN AND TEST 2 MODELS WITH KERAS CONVOLUTIONAL NEURAL NETWOEK MIRRORING OUR NEURAL NETWORK IMPLEMENTATION
# Sources:
# # https://data-flair.training/blogs/handwritten-character-recognition-neural-network/

#import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

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

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(numClasses,activation ="softmax"))

model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# predictions
history = model.fit(Xtrn, ytrnOHE, epochs=1, callbacks=[reduce_lr, early_stop],  validation_data = (Xtst,ytstOHE))
model.summary()

# scoring
print("The training accuracy is :", history.history['accuracy'])
print("The testing accuracy is :", history.history['val_accuracy'])
print("The training loss is :", history.history['loss'])
print("The testing loss is :", history.history['val_loss'])

# confusion matrices
# accuracy
# TPR and FPR
# area under ROC curve