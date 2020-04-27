#%% Dependencies

import os
import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#%% Load sample list

datapath = './data'
folderlist = os.listdir(datapath)
samples = []
for folder in folderlist:
    with open(datapath+'/'+folder+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
            
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#%% Hyperparameters

batch_size = 32
num_epochs  = 5
steering_correction = [0, 0.25, -0.25]  # for center/left/right image

#%% Data generator function

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    imagename = batch_sample[i].split('/')[-1]
                    groupname = batch_sample[i].split('/')[-3]
                    path = datapath+'/'+groupname+'/IMG/'+imagename
                    image = mpimg.imread(path)
                    
                    angle = float(batch_sample[3]) + steering_correction[i]
                    images.append(image)
                    angles.append(angle)
                    
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# generator functions for training set and validation set
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

#%% Model architecture

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#%% Train model

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=num_epochs, verbose=1)
model.save('model.h5')
