
# model.py file

import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Read in recorded data from csv file
lines = []
with open('track12/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = lines[1:]
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
shuffle(lines)
# Split recorded into training and validation sets 80:20
train_samples, valid_samples = train_test_split(lines, test_size=0.2)

#Steering correction to apply for image frames from left or right camera perspective
correction = 0.18

def augmentData(images, measurements):
    aug_images, aug_measurements = [], []
    for image, meas in zip(images, measurements):
        aug_images.append(image)
        aug_images.append(cv2.flip(image,1))
        aug_measurements.append(meas)
        aug_measurements.append(meas*-1.0)
    return (aug_images, aug_measurements)

# Batch generator to generate data during training/validation
def generator(samples, batch_size=100):
    num_samples = len(samples)
    while 1: #loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            end = offset+batch_size
            if (offset+batch_size)>num_samples:
                end = num_samples
            batch_samples = samples[offset:end]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'track12/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])

                name = 'track12/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = center_angle + correction

                name = 'track12/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = center_angle - correction

                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            X_train = np.asarray(images)
            y_train = np.asarray(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def resize(image):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(image, (66,200))
    
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.backend import tf as ktf

# Network architecture
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160,320,3), name='norm'))
# Crop image to region of interest
model.add(Cropping2D(cropping=((70,20),(0,0))))
model.add(Lambda(resize, name = 'resize'))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

def visualizeOutput(model, layer_name, image):
    layer_model = Model(input=model.input, output=model.get_layer(layer_name).output)
    image = np.expand_dims(image, axis=0)
    layer_output = layer_model.predict(image)
    print('Features shape: ', layer_output.shape)
    #plot
    plt.imshow(layer_output.squeeze())
    plt.show()

train_generator = generator(train_samples)
valid_generator = generator(valid_samples)
print('Num train samples: ', len(train_samples))
# Run training and validation model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*3, validation_data = valid_generator, nb_val_samples=len(valid_samples)*3, nb_epoch=10)

model.save('model.h5')
