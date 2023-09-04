from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LambdaCallback
from keras.layers import Input,Dropout, Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import numpy as np
import itertools 
import datetime

import cv2
import os
import io
import argparse

def t_flow(a):
    a=2+a
    a=print('{tf.version.VERSION}')
    return a

def main():
    load_dataset()
    train_model()
    predict()

def load_dataset(dataset_dir):
    global x_train,x_val,x_test,y_train,y_val,y_test,image_size
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    x_train = [] # training images.
    y_train  = [] # training labels.
    x_test = [] # testing images.
    y_test = [] # testing labels.

    image_size = 200


    for label in labels:
        trainPath = os.path.join(dataset_dir,label)
        for file in tqdm(os.listdir(trainPath)):
            image = cv2.imread(os.path.join(trainPath, file),0) # load images in gray.
            image = cv2.bilateralFilter(image, 2, 50, 50) # remove images noise.
            image = cv2.applyColorMap(image, cv2.COLORMAP_BONE) # produce a pseudocolored image.
            image = cv2.resize(image, (image_size, image_size)) # resize images into 150*150.
            x_train.append(image)
            y_train.append(labels.index(label))
        
        testPath = os.path.join('D:\ML\Dataset\MRI_DATASET\Data\Testing',label)
        for file in tqdm(os.listdir(testPath)):
            image = cv2.imread(os.path.join(testPath, file),0)
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            image = cv2.resize(image, (image_size, image_size))
            x_test.append(image)
            y_test.append(labels.index(label))


    x_train = np.array(x_train) / 255.0 # normalize Images into range 0 to 1.
    x_test = np.array(x_test) / 255.0

    print(x_train.shape)
    print(x_test.shape)

    images = [x_train[i] for i in range(15)]
    fig, axes = plt.subplots(3, 5, figsize = (10, 10))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
        
    x_train, y_train = shuffle(x_train,y_train, random_state=42) 

    y_train = tf.keras.utils.to_categorical(y_train) #One Hot Encoding on the labels
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42) #Dividing the dataset into Training and Validation sets.

    print(x_val.shape)     

    # ImageDataGenerator transforms each image in the batch by a series of random translations, rotations, etc.
    datagen = ImageDataGenerator(
        rotation_range=10,                        
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True)

    # After you have created and configured your ImageDataGenerator, you must fit it on your data.
    datagen.fit(x_train)
    return 


def train_model():
    global model
    net = ResNet50(
    weights='imagenet', # Load weights pre-trained on ImageNet.
     include_top=False, # Do not include the ImageNet classifier at the top.
     input_shape=(image_size,image_size,3))

    model = net.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.4)(model)
    model = Dense(4, activation="softmax")(model)
    model = Model(inputs= net.input, outputs= model)
    #compile our model.
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(x_train,y_train, epochs=5,batch_size=32,validation_data=(x_val,y_val))
    model.save(".\MRI_brain.keras")
    

def predict(image_dir):
    mri_image=(image_dir)
    img = tf.keras.utils.load_img(mri_image, target_size=(200, 200))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    y_pred=model.predict(img_array)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return y_pred_classes



def command_li():            
    parser=argparse.ArgumentParser(description='This app classifies mri brain tumor into gloima,meningioma and pituitary')     
    parser.add_argument("--main",help="Use --main to run the program",action="store_true") 
    args=parser.parse_args()  
    
if __name__=="__command_li__":
    command_li()    


if __name__=="__main__":
    main()   