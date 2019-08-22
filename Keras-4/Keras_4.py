import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from keras.datasets import emnist

(XTrain, yTrain), (XTest, yTest) = emnist.load_data()

XTrain = XTrain.reshape(124800,28,28,1)
XTest = XTest.reshape(20800,28,28,1)
XTrain = XTrain/255
XTest = XTest/255

YTrain = np_utils.to_categorical(yTrain,26)
YTEst = np_utils.to_categorical(yTest,26)


l2Reg = 0.001
CNN = Sequential()

CNN.add(Conv2D(32,(3,3),padding="same",activation="relu",kernel_regularizer=l2(l2Reg),input_shape=(28,28,1)))
CNN.add(MaxPool2D(pool_size=(2,2),padding="same"))

CNN.add(Conv2D(64,(3,3),padding="same",activation="relu",kernel_regularizer=l2(l2Reg)))
CNN.add(MaxPool2D(pool_size=(2,2),padding="same"))
CNN.add(Flatten())

CNN.add(Dense(80,activation="relu"))
CNN.add(Dense(40,activation="relu"))
CNN.add(Dense(10,activation="softmax"))

#CNN.load_weights("netz.h5")
CNN.summary()

CNN.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])
CNN.fit(XTrain, YTrain,epochs=2)
yP = CNN.predict(XTest)
diffCases = np.flatnonzero(np.argmax(yP,axis=1)-yTest)

print("Richtig: %.2f%% "% (100 - diffCases.shape[0]/100))

CNN.save_weights("netz.h5")
