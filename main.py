import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools


from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization


def load_mnist_data():
	# воруем картинки 28*28 пикселей и соответствующие им классы(цифры)
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	print("X_train shape", X_train.shape) # картинки для обучения сети 
	print("y_train shape", y_train.shape) # цифры изображенные на соотв. картинках для обучения
	print("X_test shape", X_test.shape) # картинки для тестирования сети
	print("y_test shape", y_test.shape) # цифры изображенные на соотв. картинках для тестирования

	# монохромная картинка представляет собой двумерный массив чисел от 0..255
	X_train = X_train.reshape(60000, 28, 28, 1)
	X_test = X_test.reshape(10000, 28, 28, 1)

	# а затем нормализовать их, сделав их "процентами"
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255 
	X_test /= 255

	# нам нужно превратить вектора цифр в вектора "категорий", указав что всего их 10
	nb_classes = 10 # number of unique digits
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, Y_train), (X_test,Y_test)

def create_model():
	# Обычная модель последовательных слоев
	model = Sequential()                                 # Linear stacking of layers

	# Convolution Layer 1
	model.add(Conv2D(32, (3, 3), input_shape=(28,28,1))) # 32 different 3x3 kernels -- so 32 feature maps
	model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
	convLayer01 = Activation('relu')                     # activation
	model.add(convLayer01)

	# Convolution Layer 2
	model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
	model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
	model.add(Activation('relu'))                        # activation
	convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
	model.add(convLayer02)

	# Convolution Layer 3
	model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
	model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
	convLayer03 = Activation('relu')                     # activation
	model.add(convLayer03)

	# Convolution Layer 4
	model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
	model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
	model.add(Activation('relu'))                        # activation
	convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
	model.add(convLayer04)
	model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

	# Fully Connected Layer 5
	model.add(Dense(512))                                # 512 FCN nodes
	model.add(BatchNormalization())                      # normalization
	model.add(Activation('relu'))                        # activation

	# Fully Connected Layer 6                       
	model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
	model.add(Dense(10))                                 # final 10 FCN nodes
	model.add(Activation('softmax'))                     # softmax activation

	# Summarize the built model
	model.summary()

	# Adam optimizer for learning
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def train_and_test_model(model, X_train, Y_train, X_test, Y_test):

	# data augmentation prevents overfitting by slightly changing the data randomly
	# Keras has a great built-in feature to do automatic augmentation

	gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
	                         height_shift_range=0.08, zoom_range=0.08)

	test_gen = ImageDataGenerator()
	# We can then feed our augmented data in batches
	# Besides loss function considerations as before, this method actually results in significant memory savings
	# because we are actually LOADING the data into the network in batches before processing each batch

	# Before the data was all loaded into memory, but then processed in batches.

	train_generator = gen.flow(X_train, Y_train, batch_size=128)
	test_generator = test_gen.flow(X_test, Y_test, batch_size=128)
	# We can now train our model which is fed data by our batch loader
	# Steps per epoch should always be total size of the set divided by the batch size

	# SIGNIFICANT MEMORY SAVINGS (important for larger, deeper networks)

	model.fit_generator(train_generator, steps_per_epoch=60000//128, epochs=5, verbose=1, 
	                    validation_data=test_generator, validation_steps=10000//128)
	
	score = model.evaluate(X_test, Y_test)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


(X_train, Y_train), (X_test, Y_test) = load_mnist_data()
model = create_model()
train_and_test_model(model, X_train, Y_train, X_test, Y_test)