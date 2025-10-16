import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
print('xtrain.shape={}'.format(xtrain.shape))
print('ytrain.shape={}'.format(ytrain.shape))
print('xtest.shape={}'.format(xtest.shape))
print('ytest.shape={}'.format(ytest.shape))
'''
xtrain.shape=(60000, 28, 28)
ytrain.shape=(60000,)
xtest.shape=(10000, 28, 28)
ytest.shape=(10000,)
'''
def plot_img(ndarr):
    img = ndarr.copy()
    img.shape = (28,28)
    plt.imshow(img, cmap='gray')
    plt.show()
plot_img(xtrain[5,:])