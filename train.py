import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import os
def inference():
	print(os.listdir("."))

	train = pd.read_csv("train.csv")
	print(train.shape)
	train.head()
	test= pd.read_csv("test.csv")
	print(test.shape)
	test.head()
	Y_train = train["label"]
	# Drop 'label' column
	X_train = train.drop(labels = ["label"],axis = 1)

	plt.figure(figsize=(15,7))
	g = sns.countplot(Y_train, palette="icefire")
	plt.title("Number of digit classes")
	Y_train.value_counts()

	X_train = X_train / 255.0
	test = test / 255.0
	print("x_train shape: ",X_train.shape)
	print("test shape: ",test.shape)

	X_train = X_train.values.reshape(-1,28,28,1)
	test = test.values.reshape(-1,28,28,1)
	print("x_train shape: ",X_train.shape)
	print("test shape: ",test.shape)


	Y_train = to_categorical(Y_train, num_classes = 10)


	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
	print("x_train shape",X_train.shape)
	print("x_test shape",X_val.shape)
	print("y_train shape",Y_train.shape)
	print("y_test shape",Y_val.shape)



	model = Sequential()
	#
	model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
		         activation ='relu', input_shape = (28,28,1)))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	#
	model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
		         activation ='relu'))
	model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	# fully connected
	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))
	optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
	epochs = 10  # for better result increase the epochs
	batch_size = 250
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # dimesion reduction
		rotation_range=5,  # randomly rotate images in the range 5 degrees
		zoom_range = 0.1, # Randomly zoom image 10%
		width_shift_range=0.1,  # randomly shift images horizontally 10%
		height_shift_range=0.1,  # randomly shift images vertically 10%
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	datagen.fit(X_train)
	history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
		                      epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
