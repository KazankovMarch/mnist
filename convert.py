# from tensorflow import keras
import tensorflow as tf
# import network as nw
# from tensorflow.contrib import lite

if __name__ == '__main__':
    # new_model= tf.keras.models.load_model(filepath="model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model_file('model.h5') 
    tfmodel = converter.convert() 
    open ("model.tflite" , "wb") .write(tfmodel)