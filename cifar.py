#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:45:58 2020
@author: sivaselvam
"""

# -*- coding: utf-8 -*-i
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
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
x_test.shape

#batches,width,height,colour 
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000,32,32,3)

#building model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()
# Convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# Convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
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
print(y_test[8],predictions[8])
