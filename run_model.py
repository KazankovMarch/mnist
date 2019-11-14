import tensorflow as tf
import pandas
import matplotlib as plt
import seaborn as sns
import data_access
import keras
import argparse
import numpy as np
from sklearn import metrics

def build_confusion_matrix(y_pred, y_test):
    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument("--uci",  action="store_true", help="use custim images from folder " + data_access.CUSTOM_IMAGES_FOLDER)
    parser.add_argument("--bcm",  action="store_true", help="build confusion matrix")
    args = parser.parse_args()
    use_custom_images_arg = args.uci
    build_confusion_matrix_arg = args.bcm


    model = keras.models.load_model(data_access.MODEL_FILE_NAME)
    if use_custom_images_arg:
        (X_test, Y_test) = data_access.load_custom_images()
    else:
        (X_train, Y_train), (X_test, Y_test) = data_access.load_mnist_data()

    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predict = model.predict(X_test)
    data_access.save_result(predict, Y_test)
    if build_confusion_matrix_arg:
        build_confusion_matrix(predict, Y_test)