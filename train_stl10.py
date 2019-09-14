import argparse

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import vgg16, mobilenet_v2
from tensorflow.python.keras.layers import Dense, BatchNormalization
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import EarlyStopping

import numpy as np
from stl10_data_loader import STL10Loader
from tools import get_logger
import skimage


def create_bb_model(learning_rate=1e-3, num_hidden_units=20, num_classes=10):
    base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling='avg', classes=num_classes)
    mu = Dense(num_hidden_units,activation='relu')(base_model.output)
    mu = BatchNormalization()(mu)
    output = Dense(num_classes, activation='softmax')(mu)
    model = Model(inputs=[base_model.input], outputs=output)
    return model


def train_stl10(y_train, y_test, x_train, x_test, num_classes, epochs, batch_size):
    model = create_bb_model(num_hidden_units=20, num_classes=num_classes)

    opt = SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc', 'mse'])
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[earlystopping])
    scores = model.evaluate(x_test, y_test, verbose=1)
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

    stl10_data_loader = STL10Loader(args.num_classes)

    (stl10_x_train, stl10_y_train, stl10_y_train_cifar10, stl10_y_train_cat), (
    stl10_x_test, stl10_y_test, stl10_y_test_cifar10, stl10_y_test_cat) = stl10_data_loader.load_dataset()


    new_shape = (32, 32, 3)
    stl10_x_train_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_train])
    stl10_x_test_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_test])
    cifar10_model = train_stl10(stl10_y_train_cat, stl10_y_test_cat, stl10_x_train_resized, stl10_x_test_resized, args.num_classes, args.epochs, args.batch_size)
    tf.keras.models.save_model(
        cifar10_model,
        args.save_dir,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
