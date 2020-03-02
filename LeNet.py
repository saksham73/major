
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import regularizers, optimizers
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, ZeroPadding2D, Flatten, Dropout
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import os
import keras
import pandas as pd
import time

# In[2]:


from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from numpy.random import permutation


# In[3]:


# Defining the four-fold LeNet model

def le_net(drop):
    model = Sequential()
    # first set of CONV => RELU => POOL
    model.add(Convolution2D(20, 5, 5, border_mode="same",
    input_shape=(60, 60, 3)))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    """
    # set of FC => RELU layers
    model.add(Dense(500))
    model.add(Dropout(drop))
    model.add(Activation("relu"))
    """
    # softmax classifier
    model.add(Dense(35))
    model.add(Dropout(drop))
    model.add(Activation("softmax"))
        
    return model 


# In[4]:

# Setting up the hyperparameters

layers = 2
drop = 0.2

split_ratio = 0.2
t_t_split = str(int(100 * (1 - split_ratio))) + '-' + str(int(100 * (split_ratio)))
epoch_count = 40


# In[4]:


# Initializing the model
model = le_net(drop)
model.summary()


# In[5]:


# Loading the numpy arrays

train_images = np.load('train_all_images_lenet_60.npy')
train_labels = np.load('train_all_labels_lenet_60.npy')
labels = train_labels


# In[6]:


# Setting up the hyper-parameters
lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('Lenet_all_marked.csv')
early_stopper = EarlyStopping(min_delta=0.001,patience=epoch_count)
model_checkpoint = ModelCheckpoint('test.h5', monitor = 'val_acc', verbose = 1, save_best_only=True)

# In[7]:


# Normalizing the data
#train_images = np.array(train_images)
#train_labels = np.array(train_labels)
mean = np.mean(train_images,axis=(0, 1, 2, 3))
std = np.std(train_images,axis=(0, 1, 2, 3))
train_images = (train_images-mean)/(std+1e-7)
num_classes = 35
train_labels = np_utils.to_categorical(train_labels,num_classes)


# In[8]:


# Train-test Split

total = len(train_images)
perm = permutation(total)
train_images = train_images[perm]
train_labels = train_labels[perm]
val_total = int(total * split_ratio)
val_images = train_images[:val_total]
val_labels = train_labels[:val_total]
new_train= train_images[(val_total ):]
new_labels = train_labels[(val_total ):]


# In[9]:


# Compiling the final model

model.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])


# In[11]:


# Realtime Dataset Augmentation Setup

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=False)


# In[12]:

"""
callbacks=[lr_reducer, csv_logger, early_stopper, model_checkpoint],
                        
"""


b_size = 24
# Training the model

start_time = time.time()
datagen.fit(new_train)
model.fit_generator(datagen.flow(new_train, new_labels, batch_size=b_size),
                        steps_per_epoch=new_train.shape[0] // b_size,
                        epochs=epoch_count, verbose=1, validation_data=(val_images,val_labels), callbacks=[csv_logger,early_stopper, model_checkpoint])
finish_time = time.time()
time_diff1 = (finish_time - start_time)/60

# In[13]:


# Save the weights of the trained model
model.save_weights('Lenet_all_60.h5')


# In[14]:


Load the weights of the trained model
model.load_weights('Lenet_all_60.h5')


# Prediction using model

class_names = ['blueberry_healthy', 'cherry_powdery_mildew', 'cherry_healthy', 'corn_cercospora_gray_leaf_spot',
             'corn_common_rust_', 'corn_northern_leaf_blight', 'corn_healthy', 'grape_black_rot', 'grape_esca',
             'grape_leaf_blight', 'grape_healthy', 'orange_haunglongbing', 'peach_bacterial_spot', 'peach_healthy',
             'pepper_bell_bacterial_spot', 'pepper_bell_healthy', 'potato_early_blight', 'potato_late_blight', 'potato_healthy',
             'raspberry_healthy', 'soybean_healthy', 'squash_powdery_mildew', 'strawberry_leaf_scorch', 'strawberry_healthy',
             'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_late_blight', 'tomato_leaf_mold', 'tomato_septoria_leaf_spot',
             'tomato_spider_mites', 'tomato_target_spot', 'tomato_yellow_leaf_curl_virus', 'tomato_mosaic_virus', 'tomato_healthy']

path_of_image = "Tomato/Tomato___Early_blight/1bb0101c-d0a4-41a5-85b9-6e8634d01a36___RS_Erly.B 9585.JPG"
img = imresize(imread(path_of_image, mode='RGB'),(60,60)).astype(np.float32)
img = (img-mean)/(std+1e-7)
img = np.expand_dims(img, axis=0)
out = model.predict(img) 
#print out
print ("class: ",np.argmax(out)," ",class_names[np.argmax(out)] )


