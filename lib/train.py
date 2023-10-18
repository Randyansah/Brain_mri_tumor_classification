from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LambdaCallback
from keras.layers import Input,Dropout, Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications.resnet import ResNet50
import keras
from keras.preprocessing.image import ImageDataGenerator

def train_model(x_train,y_train,x_val,y_val,img_size):
    datagen=ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )
    datagen.fit(x_train)
    global model
    net = ResNet50(
    weights='imagenet', # Load weights pre-trained on ImageNet.
     include_top=False, # Do not include the ImageNet classifier at the top.
     input_shape=(img_size,img_size,3))

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
    model.save("MRI_brain.keras")