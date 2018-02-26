"""
Create and save the experiments graphs
"""

import pickle 
from matplotlib import pyplot as plt

NAME = 'main-no-augm-2'
FILE = 'params_{}.bin'.format(NAME)

with open("../output/{}".format(FILE), "rb") as file:
    models = pickle.load(file)


for model in models:
    history = model['hist']
    
    # summarize history for accuracy
    acc = plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model_{0[id]} - max accuracy: {0[acc_val]:.4f}'.format(model))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    acc.savefig('../output/{}_{}_acc.png'.format(NAME, model['id']), dpi=200)
    plt.close(acc)
    
    # summarize history for loss
    loss = plt.figure()    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model_{0[id]} - min loss: {0[loss_val]:.4f}'.format(model))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    loss.savefig('../output/{}_{}_loss.png'.format(NAME, model['id']), dpi=200)    
    plt.close(loss)
