'''Data Import'''
import argparse
import json
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.layers import Dropout, Activation, Lambda
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from tools import get_logger


class Word2vecBlackBoxSelectiveNet:
    def __init__(self, input_filename, preds_filename, coverage=0.8, alpha=0.5, baseline=False, epochs=1):
        self.lamda = coverage
        self.alpha = alpha
        self.mc_dropout_rate = K.variable(value=0)
        self.epochs = epochs

        self.weight_decay = 0.0005
        self.num_classes = 2
        self._load_data(input_filename)
        self._load_preds(preds_filename)
        self.x_shape = self.x_train.shape[1]
        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        weight_decay = self.weight_decay
        basic_dropout_rate = 0.3

        model_input = Input(shape=(self.x_shape,))
        probs_mu = Input(shape=(self.num_classes,))

        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(model_input)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)
        curr = Lambda(lambda x: K.dropout(x, level=self.mc_dropout_rate))(curr)
        # classification head (f)
        curr1 = probs_mu

        # selection head (g)
        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)
        curr2 = Activation('relu')(curr2)
        curr2 = BatchNormalization()(curr2)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        curr2 = Lambda(lambda x: x / 10)(curr2)
        curr2 = Dense(1, activation='sigmoid')(curr2)
        # auxiliary head (h)
        selective_output = Concatenate(axis=1, name="selective_head")([curr1, curr2])

        auxiliary_output = probs_mu

        model = Model(inputs=[model_input, probs_mu], outputs=[selective_output])
        return model

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model.predict(x, batch_size)

    def _load_data(self, input_filename):

        # The data, shuffled and split between train, val and test sets:
        with open(input_filename, 'rb') as data_file:
            (train_X, train_y, val_X, val_y, test_X, test_y) = pickle.load(data_file)
        self.x_train = train_X
        self.x_val = val_X
        self.x_test = test_X

        self.y_train = to_categorical(np.argmax(train_y, axis=1), self.num_classes + 1)
        self.y_val = to_categorical(np.argmax(val_y, axis=1), self.num_classes + 1)
        self.y_test = to_categorical(np.argmax(test_y, axis=1), self.num_classes + 1)

    def _load_preds(self, input_filename):

        # The data, shuffled and split between train, val and test sets:
        with open(input_filename, 'rb') as data_file:
            (self.y_pred_train, self.y_pred_val, self.y_pred_test) = pickle.load(data_file)

    def test_selective_acc(self, y_true, y_pred):
        g = np.greater(y_pred[:, -1], 0.5).astype(float)
        temp1 = np.sum(
            (g) * np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred[:, :-1], axis=-1)).astype(float)
        )
        return temp1 / np.sum(g)

    def train(self):
        c = self.lamda
        lamda = 32

        def selective_loss(y_true, y_pred):
            loss = K.categorical_crossentropy(
                K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)

        # training parameters
        batch_size = 128
        maxepoches = self.epochs
        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = LearningRateScheduler(lr_scheduler)

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        self.model.compile(loss=[selective_loss], optimizer=sgd, metrics=[selective_acc, coverage])

        self.model.fit([self.x_train, self.y_pred_train], [self.y_train[:, :-1]], batch_size=batch_size,
                                epochs=maxepoches, callbacks=[reduce_lr],
                                validation_data=([self.x_val, self.y_pred_val], [self.y_val[:, :-1]]))

        # get accuracy for test
        test_pred = self.model.predict([self.x_test, self.y_pred_test])
        test_acc = self.test_selective_acc(self.y_test[:, :-1], test_pred)

        return test_acc


def train_profile(input_filename, preds_filename, coverages, alpha=0.5, epochs=1):
    results = {}
    for coverage_rate in coverages:
        logger.info("train selectivenet for {}".format(coverage_rate))
        model = Word2vecBlackBoxSelectiveNet(input_filename=input_filename,
                                             preds_filename=preds_filename,
                          coverage=coverage_rate,
                          alpha=alpha, epochs=epochs)

        acc = model.train()

        results[coverage_rate] = acc
    return results


def save_dict(filename, dict):
    with open(filename, 'w') as fp:
        json.dump(dict, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--output_results_file', type=str, default='sst2_yelp2013_results',
                        help='file to dump the results obtained')
    parser.add_argument('--input_file_name', type=str, default='data/YELP20132SST2/sst2_data.p',
                        help='file to load the data from')
    parser.add_argument('--preds_file_name', type=str, default='data/YELP20132SST2/sst2_yelp2013_preds_data.p',
                        help='file to load the data from')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weigth of the selective loss')

    args = parser.parse_args()
    input_file_name = args.input_file_name
    preds_file_name = args.preds_file_name
    epochs = args.epochs
    output_results_file = args.output_results_file
    alpha = args.alpha
    logger = get_logger()

    #coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    coverages = [0.95]
    logger.info("train selectivenet")
    results = train_profile(input_file_name, preds_file_name, coverages)
    save_dict(output_results_file, results)
