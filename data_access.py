import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers
import data_access

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools


from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization
import cv2
import numpy as np
from keras.datasets import mnist
from scipy import misc
import json, os


CUSTOM_IMAGES_FOLDER =  'images'

MODEL_FILE_NAME = "model.h5"

nb_classes = 10 # number of unique digits
    
def load_mnist_data():
    # load the MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #  do some formatting
    # Except we do not flatten each image into a 784-length vector because we want to perform convolutions first

    X_train = X_train.reshape(60000, 28, 28, 1) #add an additional dimension to represent the single-channel
    X_test = X_test.reshape(10000, 28, 28, 1)

    X_train = X_train.astype('float32')         # change integers to 32-bit floating point numbers
    X_test = X_test.astype('float32')

    X_train /= 255                              # normalize each value for each pixel for the entire vector for each input
    X_test /= 255

    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)
    # one-hot format classes


    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return (X_train, Y_train), (X_test, Y_test)


def save_result(predict, result):
    data = []
    for i in range(len(predict)):
        res = {
            'Real:': str(np.argmax(result[i])),
            'Await': str(predict[i]),
        }
        data.append(res)

    with open('result.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# загрузка собственны изображений из папки
def load_custom_images():
    X_test = []
    Y_test = []
    count = 0
    for file in os.listdir(CUSTOM_IMAGES_FOLDER):
        imgarr = cv2.imread(CUSTOM_IMAGES_FOLDER +'/' + file, cv2.IMREAD_GRAYSCALE)
        X_test.append(imgarr)
        Y_test.append(int(file.split('.')[0]))
        count = count + 1
    X_test =  np.array(X_test)
    X_test = X_test.reshape(len(X_test), 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    return X_test, Y_test
