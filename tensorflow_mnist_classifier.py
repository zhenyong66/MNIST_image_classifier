import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#%% load & process data
(train_img, train_label), (test_img, test_label) = mnist.load_data()
train_img = train_img.astype('float32') / 255.
test_img = test_img.astype('float32') / 255.
train_img = np.reshape(train_img, (len(train_img), 28, 28, 1))
test_img = np.reshape(test_img, (len(test_img), 28, 28, 1))
print('Train images shape:', train_img.shape)
print('Test images shape:', test_img.shape)

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

val_img = train_img[50000:]
val_label = train_label[50000:]


#%% build model
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(8, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(4, (1,1), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#%% train model
history = model.fit(train_img, train_label,
                    epochs = 10,
                    batch_size = 128,
                    shuffle = True,
                    validation_data = (val_img, val_label))

#%% predictions for the first 10 test images
prob = model.predict(test_img)
pred = []

for i in range(9):    
    pred.append(np.argmax(prob[i]))

print(pred)

#%% display the first 10 images
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    ax = plt.subplot(2, n, i)
    plt.imshow(test_img[i-1].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()