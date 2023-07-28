import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import random
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im
from PIL import ImageDraw as imDraw
from image_modules import *
import tensorflow as tf
import keras
import time
from sklearn.metrics import accuracy_score
from custom_callback import Save_Accuracy_By_Epoch, Save_Multiclass_Metrics_By_Epoch

# Find accuracy of model
def find_accuracy(test,pred):
    correct = 0
    total = len(test)

    for i in range(len(test)):
        if test[i] == pred[i]:
            correct += 1

    return correct/total 


# Map ANN outputs to classes
def get_labels(y_pred_ann): 
    labels = []

    for pred in y_pred_ann:
        max_index = 0

        for i in range(len(pred)):
            if pred[i] > pred[max_index]:
                max_index = i
        
        labels.append(max_index)

    return labels

# Read Train images from Directory
def read_images_from_path(img_dir):
    image_paths = []
    
    for image_name in os.listdir(img_dir):
        image_paths.append(img_dir.join(['',image_name]))

    images = []
    
    counter = 0
    for path in image_paths:
        img = mpimg.imread(path)
        img = tf.image.convert_image_dtype(img, tf.float32)
            
        # Resizing image
        img = tf.image.resize(img, (128,128))

        images.append(img.numpy())
        
        counter += 1

        if counter % 100 == 0:
            print(str(counter) + " images read.")
        
    return np.array(images) 

def read_labels(path_train, path_test):
    # Read Train Labels from directory
    labels_pd_train = pd.read_csv(path_train)
    labels_train = labels_pd_train.to_numpy()
    labels_train  = labels_train[labels_train[:,0].argsort()]
    labels_train = labels_train[:,1:]
    labels_train = labels_train.flatten()

    # Read Test Labels from directory
    labels_pd_test = pd.read_csv(path_test)
    labels_test = labels_pd_test.to_numpy()
    labels_test  = labels_test[labels_test[:,0].argsort()]
    labels_test = labels_test[:,1:]
    labels_test = labels_test.flatten()
    
    return labels_train, labels_test




train_img_path = 'F:\\FYP\\Initial\\apples\\train\\'
test_img_path = 'F:\\FYP\\Initial\\apples\\test\\'
train_label_path = 'F:/FYP/Initial/apples/trian.csv'
test_label_path = 'F:/FYP/Initial/apples/test.csv'


train_labels, test_labels = read_labels(train_label_path, test_label_path)
train_images_raw = read_images_from_path(train_img_path)
test_images_raw = read_images_from_path(test_img_path)

train_images = []
for i in range(len(train_images_raw)):
    train_images.append(train_images_raw[i].flatten())
    pass

test_images = []
for i in range(len(test_images_raw)):
    test_images.append(test_images_raw[i].flatten())
    pass

train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = train_labels.astype('float32')
test_labels = test_labels.astype('float32')



train_images = np.asarray(train_images).astype('float32')
test_images = np.asarray(test_images).astype('float32')


tf.keras.utils.set_random_seed(200)
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(5,5),activation = "relu" , input_shape = (128,128,3)) ,
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(5,5),activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(4,4),activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),activation = "relu"),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(550,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.5,seed = 2019),
        tf.keras.layers.Dense(400,activation ="relu"),
        tf.keras.layers.Dropout(0.2,seed = 2019),
        tf.keras.layers.Dense(10,activation = "softmax")   #Adding the Output Layer
    ])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['acc'])

model.fit(train_images_raw, train_labels, epochs=100, callbacks=[Save_Multiclass_Metrics_By_Epoch((test_images_raw,test_labels),6)])

pass