#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:31:23 2020

@author: Siva-selvam
"""

# -*- coding: utf-8 -*-i
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0])

#to categorical everything for labels
y_cat_test = to_categorical(y_test)
y_cat_train = to_categorical(y_train)
print(y_cat_test)

#normalizing the test data
print(x_test.max())
print(x_test.min())

x_train = x_train/255
x_test = x_test/255
plt.imshow(x_train[0])

#batches,width,height,colour is 1 i.e greyscale
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

#building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
# Convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# flatten images from 28 by 28 to 764 before finallayer
model.add(Flatten())
# 128 Neurons in Dense hiddenlayer
model.add(Dense(128, activation='relu'))
# classifier
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#earlystoopint to prevent overfit
early_stopping = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stopping])

losses = pd.DataFrame(model.history.history)
losses[['accuracy','val_accuracy']].plot()

predictions = model.predict_classes(x_test)
print(model.evaluate(x_test,y_cat_test,verbose=0))
print(model.metrics_names)
print(classification_report(y_test,predictions))
print(y_test[0],predictions[0])
