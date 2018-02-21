"""
VGG19 with the combination of the parameters:
- 'hdd_size': [512, 1024]
- 'dr': [0.1, 0.3]
- 'lr': [0.0001]
- 'bsz': [64]
- 'deep': [1, 2, 4]
- 'act_fun': ['relu']

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
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm

# import my library
# sys.path.append('../notebook/my_lib/')
# from data_augmentation import DataAugmentation

# import data
csv_train = pd.read_csv('../input/labels.csv')

# DEBUG: reduce data for test
# csv_train = csv_train.head(1000)

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

# Data argumentation
# data_aug = DataAugmentation(x_train, options={'horizontal_flips': True,
#                                               'rotation': True,
#                                               'rotation_config': [(20,1.3)]})
# for i, images in enumerate(tqdm(data_aug)):
#     for image in images:
#         # if i == 4:
#         #     plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
#         #     plt.show()
#         x_train.append(image)
#         y_train.append(y_train[i])


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
                                                      test_size=0.05, random_state=42,
                                                      stratify=y_train_raw)


def model_builder(hdd_size=128, dr= 0.1, learning_rate= 0.003, act_fun = 'relu', deep=5):
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(im_size, im_size, 3))

    # Add a new top layers
    x = base_model.output
    x = Flatten()(x)
    for _ in range(deep):
        x = Dense(hdd_size, activation=act_fun)(x)
        x = Dropout(dr)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
        
    optimizer = optimizers.Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy'])

    callbacks_list = [
        # keras.callbacks.ModelCheckpoint('../output/model_' + NAME + '_{epoch:02d}-{val_loss:.2f}.h5',
        #                                 monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
        #                                 mode='auto', period=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
    model.summary()
    return model, callbacks_list


param_grid = {'hdd_size': [512, 1024],
              'dr': [0.1, 0.3],
              'lr': [0.0001],
              'bsz': [64],
              'deep': [1, 2, 4],
              'act_fun': ['relu']}

grid = list(ParameterGrid(param_grid))
pprint(grid)
input()

grind_ris = []
best_loss = 10000
for i, param in enumerate(grid):
    model, callbacks_list = model_builder(hdd_size=param['hdd_size'],
                                          dr=param['dr'],
                                          learning_rate=param['lr'],
                                          act_fun=param['act_fun'],
                                          deep=param['deep'])

    history = model.fit(X_train, Y_train, epochs=200, batch_size=param['bsz'],
                        validation_data=(X_valid, Y_valid),
                        callbacks=callbacks_list, verbose=1)
    loss, acc = min(history.history['val_loss']), max(history.history['val_acc'])

    h = {k: history.history[k] for k in history.history.keys()}

    grind_ris.append({'id': i, 'loss_val': loss, 'acc_val': acc, 'par': param, 'hist': h})

    if loss < best_loss:
        print('Found a better loss!!')
        best_loss = loss
        model.save('../output/bestmodel_{}-{}_{:.2f}.h5'.format(NAME, i, loss))

    
    ## salva su file la grind_ris usando pickle
    pickle.dump(grind_ris, open('../output/params_{}.bin'.format(NAME), 'wb'))
