from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical
import numpy as np


class Cifar10Loader(object):

    def __init__(self, num_classes):
        assert num_classes in [2, 10]
        self.num_classes = num_classes

    def get_cifar10_data(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        if self.num_classes == 2:
            # filter classes 0 and 1
            x_train = x_train[np.isin(y_train.flatten(), [0, 1])]
            y_train = y_train[np.isin(y_train.flatten(), [0, 1])]
            x_test = x_test[np.isin(y_test.flatten(), [0, 1])]
            y_test = y_test[np.isin(y_test.flatten(), [0, 1])]
        cifar10_y_train = to_categorical(y_train, self.num_classes)
        cifar10_y_test = to_categorical(y_test, self.num_classes)
        cifar10_x_train = x_train.astype('float32')
        cifar10_x_test = x_test.astype('float32')
        cifar10_x_train /= 255.0
        cifar10_x_test /= 255.0
        return cifar10_y_train, cifar10_y_test, cifar10_x_train, cifar10_x_test
