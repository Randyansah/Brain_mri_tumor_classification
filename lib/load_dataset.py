import tensorflow as tf
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
tf.distribute.MirroredStrategy(gpus)


def load_dataset(train_dir,test_dir):
    global x_train,x_val,x_test,y_train,y_val,y_test,image_size
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    x_train = [] # training images.
    y_train  = [] # training labels.
    x_test = [] # testing images.
    y_test = [] # testing labels.

    image_size = 200


    for label in labels:
        trainPath = os.path.join(train_dir,label)
        for file in tqdm(os.listdir(trainPath)):
            image = cv2.imread(os.path.join(trainPath, file),0) # load images in gray.
            image = cv2.bilateralFilter(image, 2, 50, 50) # remove images noise.
            image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) # produce a pseudocolored image.
            image = cv2.resize(image, (image_size, image_size)) # resize images into 150*150.
            x_train.append(image)
            y_train.append(labels.index(label))
        
        testPath = os.path.join(test_dir,label)
        for file in tqdm(os.listdir(testPath)):
            image = cv2.imread(os.path.join(testPath, file),0)
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            image = cv2.resize(image, (image_size, image_size))
            x_test.append(image)
            y_test.append(labels.index(label))
    x_train = np.array(x_train) / 255.0 # normalize Images into range 0 to 1.
    x_test = np.array(x_test) / 255.0      
    x_train, y_train = shuffle(x_train,y_train, random_state=42) 

    y_train = tf.keras.utils.to_categorical(y_train) #One Hot Encoding on the labels
    y_test = tf.keras.utils.to_categorical(y_test)  
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42) #Dividing the dataset into Training and Validation sets.
    return x_train,x_val,x_test,y_train,y_val,y_test
