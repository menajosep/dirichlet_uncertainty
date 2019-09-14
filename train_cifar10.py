import argparse

import tensorflow as tf
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.optimizers import SGD

from cifar_data_loader import Cifar10Loader
from tools import get_logger


def create_vgg16(num_classes):
    model = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=num_classes)
    return model


def train_cifar10(cifar10_y_train, cifar10_y_test, cifar10_x_train, cifar10_x_test, num_classes, epochs, batch_size):
    model = create_vgg16(num_classes)
    opt = SGD(lr=1e-2, momentum=0.9, decay=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])
    model.fit(cifar10_x_train, cifar10_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True)
    scores = model.evaluate(cifar10_x_test, cifar10_y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs to train original model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='dir to save the trained model')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes for the trained model')

    args = parser.parse_args()
    logger = get_logger()

    cifar10_data_loader = Cifar10Loader(args.num_classes)

    cifar10_y_train, cifar10_y_test, cifar10_x_train, cifar10_x_test = cifar10_data_loader.get_cifar10_data()
    cifar10_model = train_cifar10(cifar10_y_train, cifar10_y_test, cifar10_x_train, cifar10_x_test, args.num_classes, args.epochs, args.batch_size)
    tf.keras.models.save_model(
        cifar10_model,
        args.save_dir,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )