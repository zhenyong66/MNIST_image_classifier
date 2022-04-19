import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#%% load & process data
(train_img, train_label), (test_img, test_label) = cifar10.load_data()
train_img = train_img.astype('float32') / 255.
test_img = test_img.astype('float32') / 255.
train_img = np.reshape(train_img, (len(train_img), 32, 32, 3))
test_img = np.reshape(test_img, (len(test_img), 32, 32, 3))

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

val_img = train_img[40000:]
val_label = train_label[40000:]

print('Train images shape:', train_img.shape)
print('Validation images shape:', val_img.shape)
print('Test images shape:', test_img.shape)

# plt.figure(figsize=(20, 4))
# for i in range(1, 10):
#     ax = plt.subplot(1, 10, i)
#     plt.imshow(train_img[i-1].reshape(32, 32, 3))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

#%% build model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding = 'same', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, (3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(8, (3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#%% train model
history = model.fit(train_img, train_label,
                    epochs = 100,
                    batch_size = 128,
                    shuffle = True,
                    validation_data = (val_img, val_label))

#%% predictions for the first 10 test images
labels = {0: 'airplane',
          1: 'automobile',
          2: 'bird',
          3: 'cat',
          4: 'deer',
          5: 'dog',
          6: 'frog',
          7: 'horse',
          8: 'ship',
          9: 'truck'}

prob = model.predict(test_img)
pred = []

for i in range(test_label.shape[0]):    
    pred.append(np.argmax(prob[i]))

correct = 0
for i in range(test_label.shape[0]):    
    if np.argmax(prob[i]) == np.argmax(test_label[i]):
        correct += 1
        
test_acc = correct / test_label.shape[0]
print('Accuracy:', str(test_acc*100) + '%')

#%% display the first 10 images
# plt.figure(figsize=(20, 4))
# for i in range(1, 10):
#     ax = plt.subplot(1, 10, i)
#     plt.imshow(test_img[i-1].reshape(32, 32, 3))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()