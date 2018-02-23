"""
VGG16 base-line

10% validation

optimizer Adam

image size 90

No augmentation
"""
NAME = __file__.split('.')[0]

import pickle
import random
import sys
from pprint import pprint

import cv2
import keras
import numpy as np
import pandas as pd
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm

# import data
csv_train = pd.read_csv('../input/labels.csv')

# Generate Labels
targets_series = pd.Series(csv_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)
labels = np.asarray(one_hot)

im_size = 90

x_train = []
y_train = []

for i, (f, breed) in enumerate(tqdm(csv_train.values)):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(labels[i])

# build np array and normalise them
x_train_raw = np.array(x_train, np.float32) / 255.
y_train_raw = np.array(y_train, np.uint8)
print("x_train shape:", x_train_raw.shape)
print("y_train shape:", y_train_raw.shape)
input()

# usesfull variable
num_classes = y_train_raw.shape[1]

# Using the stratify parameter on treain_test_split the split should be equally distributed per classes.
# Try a small percentage of dataset for validation (5%)
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw,
                                                      test_size=0.10, random_state=42,
                                                      stratify=y_train_raw)


base_model = VGG16(weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layers
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

callbacks_list = [
    keras.callbacks.ModelCheckpoint('../output/model_' + NAME + '_{epoch:02d}-{val_loss:.2f}.h5',
                                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
                                    mode='auto', period=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)]
model.summary()
input()

# Train the model
history = model.fit(X_train, Y_train, epochs=200, validation_data=(X_valid, Y_valid),
                    callbacks=callbacks_list, verbose=1)

# save model statistics using pickle
min_loss, max_acc = min(history.history['val_loss']), max(history.history['val_acc'])
history_plot = {k: history.history[k] for k in history.history.keys()}
grind_ris = {'loss_val': min_loss, 'acc_val': max_acc, 'hist': history_plot}
pickle.dump(grind_ris, open('../output/params_{}.bin'.format(NAME), 'wb'))
