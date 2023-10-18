import tensorflow as tf
import os
import numpy as np

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
tf.distribute.MirroredStrategy(gpus)

def predict(model,dir,filename):
    image_dir=os.path.join(dir,filename)
    mri_image=(image_dir)
    img = tf.keras.utils.load_img(mri_image, target_size=(200, 200))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    y_pred=model.predict(img_array)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return y_pred_classes