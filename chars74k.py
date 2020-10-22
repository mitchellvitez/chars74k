"""
Convolutional network that does real-world character recognition
Mitchell Vitez 2020

Input: images of characters from the chars74k dataset
Output: a character in the set "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

Dataset from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
"""

import glob
import numpy as np
import random
from cv2 import resize
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

SAMPLES = 7700
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ROWS = 18
COLS = 12

images = np.zeros(shape=(SAMPLES, ROWS, COLS, 3))
labels = np.zeros(shape=(SAMPLES,), dtype=int)
sample_num = 0
for sample in sorted(glob.glob('data/chars74k/Img/GoodImg/Bmp/*')):
    for image in glob.glob(sample + '/*'):
        images[sample_num] = resize(imread(image), dsize=(COLS, ROWS))
        labels[sample_num] = int(sample[-2:]) - 1
        sample_num += 1

images, images_test, labels, labels_test = train_test_split(images, labels, test_size=0.2)

print(f'{images.shape[0]} sample images of size {images.shape[1]}x{images.shape[2]}')
print(f'{labels.shape[0]} labels')
assert images.shape[0] == labels.shape[0]

layers = \
  [ Convolution2D(128, 3, 3, input_shape=(ROWS, COLS, 3), activation='relu')
  , Convolution2D(256, 2, 1, activation='relu')
  , Convolution2D(512, 2, 1, activation='relu')
  , MaxPooling2D(pool_size=(2, 2))
  , Flatten()
  , Dense(1024, activation='relu')
  , Dropout(0.5)
  , Dense(512, activation='relu')
  , Dropout(0.5)
  , Dense(62)
  , Activation('softmax')
  ]

model = Sequential()
for layer in layers:
    model.add(layer)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, labels, epochs=20, batch_size=128, validation_data=(images_test, labels_test))
model.summary()

predictions = model.predict(images_test)
for _ in range(3):
    i = random.randint(0, images_test.shape[0])
    print('\nselected a random image')
    imshow(images_test[i])
    print('prediction:', CHARS[np.argmax(predictions[i])])
    chance = predictions[i][np.argmax(predictions[i])]
    print('confidence: {:0.2f}%'.format(chance * 100))
    print('actual:', CHARS[labels_test[i]])
