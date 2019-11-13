from tensorflow import keras
import data_access
from keras.utils import to_categorical
import argparse

def build_confusion_matrix(predict, y, count):

    predict = list(map(np.argmax, predict) )
    y = list(map(np.argmax, y) )
    matrix = tensorflow.math.confusion_matrix(labels=y, predictions=predict).numpy()
    matrix = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    columns = []
    for i in range(0, count):
        columns.append(i)

    frame = pandas.DataFrame(matrix,
                         index = columns,
                         columns = columns)

    seaborn.heatmap(frame, annot=True,cmap=matplotlib.pyplot.cm.Blues)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted label')
    matplotlib.pyplot.show()



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
        print("RYAAAAAAAAAAAAAAAAAAAAAAAA")
        (X_train, Y_train), (X_test, Y_test) = data_access.load_mnist_data()

    print('4=========++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=====')
    score = model.evaluate(X_test, Y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print('5========++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++======')
    predict = model.predict(X_test)
    print('6=======++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=====')
    data_access.save_result(predict, Y_test)
    print('7=======++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++======')
    if build_confusion_matrix_arg:
        build_confusion_matrix(predict, Y_test, 10)