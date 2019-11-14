'''Data Import'''
import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.python.keras.applications import mobilenet_v2
import tensorflow as tf
import tensorflow_probability as tfp

from tools import get_logger


class cifar10vgg:
    def __init__(self, coverage=0.8, alpha=0.5, baseline=False):
        self.lamda = coverage
        self.alpha = alpha
        self.mc_dropout_rate = K.variable(value=0)
        self.num_classes = 10
        self.weight_decay = 0.0005
        self._load_data()

        self.x_shape = self.x_train.shape[1:]

        self.model = self.build_model()
        if baseline:
            self.alpha = 0

        self.model = self.train(self.model)

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        weight_decay = self.weight_decay
        basic_dropout_rate = 0.3
        model_input = Input(shape=self.x_shape)
        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(model_input)
        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate)(curr)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr = Flatten()(curr)
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)
        curr = Lambda(lambda x: K.dropout(x, level=self.mc_dropout_rate))(curr)

        # classification head (f)
        curr1 = Dense(self.num_classes, activation='softmax')(curr)

        # selection head (g)
        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)
        curr2 = Activation('relu')(curr2)
        curr2 = BatchNormalization()(curr2)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        curr2 = Lambda(lambda x: x / 10)(curr2)
        curr2 = Dense(1, activation='sigmoid')(curr2)
        # auxiliary head (h)
        selective_output = Concatenate(axis=1, name="selective_head")([curr1, curr2])

        auxiliary_output = Dense(self.num_classes, activation='softmax', name="classification_head")(curr)

        model = Model(inputs=model_input, outputs=[selective_output, auxiliary_output])

        self.model_input = model_input
        self.model_embeding = Model(inputs=model_input, outputs=curr)
        return model

    def normalize(self, X_train, X_val, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_val = (X_val - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_val, X_test

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model.predict(x, batch_size)

    def predict_embedding(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model_embeding.predict(x, batch_size)

    def mc_dropout(self, batch_size=1000, dropout=0.5, iter=100):
        K.set_value(self.mc_dropout_rate, dropout)
        repititions = []
        for i in range(iter):
            _, pred = self.model.predict(self.x_test, batch_size)
            repititions.append(pred)
        K.set_value(self.mc_dropout_rate, 0)

        repititions = np.array(repititions)
        mc = np.var(repititions, 0)
        mc = np.mean(mc, -1)
        return -mc

    def selective_risk_at_coverage(self, coverage, mc=False, wrapper=False, uncertainties=None):
        _, pred = self.predict()

        if mc:
            #sr = np.max(pred, 1)
            sr = self.mc_dropout()
        elif wrapper:
            sr = -uncertainties
        else:
            sr = np.max(pred, 1)
            #sr = self.mc_dropout()
        sr_sorted = np.sort(sr)
        threshold = sr_sorted[pred.shape[0] - int(coverage * pred.shape[0])]
        covered_idx = sr > threshold
        selective_acc = np.mean(np.argmax(pred[covered_idx], 1) == np.argmax(self.y_test[covered_idx], 1))
        return selective_acc

    def _load_data(self):
        # The data, shuffled and split between train and test sets:
        (X, y), (x_test, y_test_label) = cifar10.load_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        x_train = X_train.astype('float32')
        x_val = X_val.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train, self.x_val, self.x_test  = self.normalize(x_train, x_val, x_test)
        self.y_train = to_categorical(y_train, self.num_classes + 1)
        self.y_val = to_categorical(y_val, self.num_classes + 1)
        self.y_test = to_categorical(y_test_label, self.num_classes + 1)

    def train(self, model):
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
        #maxepoches = 300
        maxepoches = 1
        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = LearningRateScheduler(lr_scheduler)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        model.compile(loss=[selective_loss, 'categorical_crossentropy'], loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc, coverage])

        historytemp = model.fit([self.x_train], [self.y_train, self.y_train[:, :-1]], batch_size=batch_size,
                              epochs=maxepoches, callbacks=[reduce_lr],
                              validation_split=0.2)
        return model


class UncertaintyWrapper(object):
    def __init__(self, lambda_reg, num_samples,
                 learning_rate=1e-3, num_hidden_units=20):
        self.lambda_reg = lambda_reg
        self.epsilon = 1e-10
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.num_hidden_units = num_hidden_units
        self.num_classes = 10

    def dirichlet_aleatoric_cross_entropy(self, y_true, y_pred):
        mu_probs = y_pred[:, :self.num_classes]
        logits_sigma = y_pred[:, self.num_classes:]
        beta = logits_sigma
        alpha = mu_probs * beta
        dirichlet = tfp.distributions.Dirichlet(alpha)
        z = dirichlet.sample(sample_shape=self.num_samples)
        e_probs = tf.reduce_mean(z, axis=0)
        log_probs = tf.log(e_probs + self.epsilon)
        cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
        return cross_entropy + self.lambda_reg * tf.reduce_sum(beta, axis=-1)

    # metric that outputs the max/min value for the beta
    def max_beta(self, y_true, y_pred, **args):
        logits_psi = y_pred[:, self.num_classes:]
        return tf.reduce_max(logits_psi)

    def min_beta(self, y_true, y_pred, **args):
        logits_psi = y_pred[:, self.num_classes:]
        return tf.reduce_min(logits_psi)

    # metric that outputs the accuracy when only considering the logits_mu.
    # this accuracy should be the same that was obtained with the fake classifier
    # in its best epoch.
    # def mu_accuracy(self):
    #  num_classes = self.num_classes
    def mu_accuracy(self, y_true, y_pred, **args):
        logits_phi = y_pred[:, :self.num_classes]
        labels_phi = y_true[:, :self.num_classes]
        return categorical_accuracy(labels_phi, logits_phi)

    #  return get_mu_accuracy

    def create_model(self, input_shape):
        base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=(32, 32, 3), pooling='avg', classes=self.num_classes)
        beta = base_model.output
        beta = Dense(self.num_hidden_units, activation='relu')(beta)
        beta = Dense(self.num_hidden_units, activation='relu')(beta)
        beta = Dense(self.num_hidden_units, activation='relu')(beta)
        beta = Dense(1, activation='softplus')(beta)
        mu_input = Input(shape=(self.num_classes,))
        output = concatenate([mu_input, beta])

        model = Model(inputs=[base_model.input, mu_input], outputs=output)
        model.compile(loss=self.dirichlet_aleatoric_cross_entropy,
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=[self.max_beta, self.min_beta]
                      )
        return model

    def train_model(self, X, y, pred_y, batch_size=256, epochs=50):
        self.num_classes = pred_y.shape[1]
        input_shape = X.shape[1]
        self.model = self.create_model(input_shape)
        self.training_history = self.model.fit([X, pred_y],
                                               y,
                                               batch_size=batch_size,
                                               epochs=epochs,
                                               shuffle=True,
                                               verbose=1,
                                               validation_split=0.2)

    def predict_entropy(self, X, pred_y):
        y_pred = self.model.predict([X, pred_y])
        logits_mu = y_pred[:, :self.num_classes]
        mu_probs = tf.nn.softmax(logits_mu, axis=-1)
        logits_sigma = y_pred[:, self.num_classes:]
        beta = logits_sigma
        alpha = mu_probs * logits_sigma
        dirichlet = tfp.distributions.Dirichlet(alpha)
        z = dirichlet.sample(sample_shape=self.num_samples)
        e_probs = tf.reduce_mean(z, axis=0)
        log_probs = tf.log(e_probs + self.epsilon)
        entropy = tf.reduce_sum(-e_probs * log_probs, axis=-1)
        return entropy


import json


def save_dict(filename, dict):
    with open(filename, 'w') as fp:
        json.dump(dict, fp)


def calc_selective_risk(model, calibrated_coverage=None):
    prediction, pred = model.predict()
    if calibrated_coverage is None:
        threshold = 0.5
    else:
        threshold = np.percentile(prediction[:, -1], 100 - 100 * calibrated_coverage)
    covered_idx = prediction[:, -1] > threshold

    coverage = np.mean(covered_idx)
    y_hat = np.argmax(prediction[:, :-1], 1)
    loss = np.sum(y_hat[covered_idx] != np.argmax(model.y_test[covered_idx, :], 1)) / np.sum(covered_idx)
    return loss, coverage


def train_profile(model_name, coverages, model_baseline=None, uncertainties=None, alpha=0.5):
    results = {}
    for coverage_rate in coverages:
        model = cifar10vgg(coverage=coverage_rate, alpha=alpha)

        loss, coverage = calc_selective_risk(model)

        results[coverage] = {"lambda": coverage_rate, "selective_risk": loss}
        if model_baseline is not None:
            results[coverage]["baseline_sr_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["percentage_sr"] = 1 - results[coverage]["selective_risk"] / results[coverage][
                "baseline_sr_risk"]
            results[coverage]["baseline_mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))
            results[coverage]["percentage_mc"] = 1 - results[coverage]["selective_risk"] / results[coverage][
                "baseline_mc_risk"]
            if uncertainties is not None:
                results[coverage]["baseline_unc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage,
                                                                                                        wrapper=True,
                                                                                                        uncertainties=uncertainties))
                results[coverage]["percentage_unc"] = 1 - results[coverage]["selective_risk"] / results[coverage][
                    "baseline_unc_risk"]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--num_units', type=int, default=40,
                        help='num units')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--output_results_file', type=str, default='cifar10_results',
                        help='file to dump the results obtained')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weigth of the selective loss')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='number of MC samples')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                        help='Lambda parameter for regularization of beta values')

    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_units = args.num_units
    batch_size = args.batch_size
    output_results_file = args.output_results_file
    alpha = args.alpha
    lambda_reg = args.lambda_reg
    num_samples = args.num_samples
    logger = get_logger()
    cifar10_model_baseline = cifar10vgg(baseline=True)
    cifar10_val_y_pred = cifar10_model_baseline.predict(cifar10_model_baseline.x_val)[1]
    cifar10_test_y_pred = cifar10_model_baseline.predict(cifar10_model_baseline.x_test)[1]
    cifar10_wrapper = UncertaintyWrapper(lambda_reg, num_samples, learning_rate=learning_rate, num_hidden_units=num_units)
    cifar10_wrapper.train_model(cifar10_model_baseline.x_val, cifar10_model_baseline.y_val[:,:-1], cifar10_val_y_pred, epochs=epochs, batch_size=batch_size)
    cifar10_test_uncertainties = K.get_session().run(cifar10_wrapper.predict_entropy(cifar10_model_baseline.x_test, cifar10_test_y_pred))
    coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    model_name = 'cifar10_selectivenet'
    results = train_profile(model_name, coverages, model_baseline=cifar10_model_baseline, alpha=alpha, uncertainties=cifar10_test_uncertainties)
    save_dict("{}.json".format(model_name), results)