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


class Word2vecSelectiveNet:
    def __init__(self, input_filename, coverage=0.8, alpha=0.5, baseline=False, epochs=1):
        self.lamda = coverage
        self.alpha = alpha
        self.mc_dropout_rate = K.variable(value=0)
        self.epochs = epochs

        self.weight_decay = 0.0005
        self.num_classes = 2
        self._load_data(input_filename)
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
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(model_input)

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
        return model

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model.predict(x, batch_size)

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
            # sr = np.max(pred, 1)
            sr = self.mc_dropout()
        elif wrapper:
            sr = -uncertainties
        else:
            sr = np.max(pred, 1)
            # sr = self.mc_dropout()
        sr_sorted = np.sort(sr)
        threshold = sr_sorted[pred.shape[0] - int(coverage * pred.shape[0])]
        covered_idx = sr > threshold
        selective_acc = np.mean(np.argmax(pred[covered_idx], 1) == np.argmax(self.y_test[covered_idx], 1))
        return selective_acc

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
        maxepoches = self.epochs
        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = LearningRateScheduler(lr_scheduler)

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
        model_input = Input(shape=(input_shape,))
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(model_input)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(self.num_hidden_units, activation='relu')(logits_sigma)
        logits_sigma = Dense(1, activation='softplus')(logits_sigma)
        probs_mu = Input(shape=(self.num_classes,))
        output = concatenate([probs_mu, logits_sigma])

        model = Model(inputs=[model_input, probs_mu], outputs=output)
        model.compile(loss=self.dirichlet_aleatoric_cross_entropy,
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=[self.mu_accuracy, self.min_beta, self.max_beta])
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
        alpha = mu_probs * beta
        dirichlet = tfp.distributions.Dirichlet(alpha)
        z = dirichlet.sample(sample_shape=self.num_samples)
        e_probs = tf.reduce_mean(z, axis=0)
        log_probs = tf.log(e_probs + self.epsilon)
        entropy = tf.reduce_sum(-e_probs * log_probs, axis=-1)
        return entropy


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


def train_profile(input_filename, coverages, model_baseline=None, uncertainties=None, alpha=0.5, epochs=1):
    results = {}
    for coverage_rate in coverages:
        model = Word2vecSelectiveNet(input_filename=input_filename,
                          coverage=coverage_rate,
                          alpha=alpha,epochs=epochs)

        loss, coverage = calc_selective_risk(model)

        results[coverage] = {"lambda": coverage_rate, "selective_risk": loss}
        if model_baseline is not None:
            results[coverage]["baseline_sr_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["percentage_sr"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_sr_risk"]
            results[coverage]["baseline_mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))
            results[coverage]["percentage_mc"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_mc_risk"]
            if uncertainties is not None:
                results[coverage]["baseline_unc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage,
                                                                                                        wrapper=True,
                                                                                                        uncertainties=uncertainties))
                results[coverage]["percentage_unc"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_unc_risk"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--seletivenet_epochs', type=int, default=1,
                        help='epochs to seletivenet model')
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
    input_file_name = args.input_file_name
    epochs = args.epochs
    seletivenet_epochs = args.seletivenet_epochs
    learning_rate = args.learning_rate
    num_units = args.num_units
    batch_size = args.batch_size
    output_results_file = args.output_results_file
    alpha = args.alpha
    lambda_reg = args.lambda_reg
    num_samples = args.num_samples
    logger = get_logger()
    logger.info("create baseline")
    sst2_model_baseline = Word2vecSelectiveNet(input_filename=input_file_name,
                                               baseline=True,
                                               epochs=seletivenet_epochs)
    logger.info("predict val with baseline")
    sst2_val_y_pred = sst2_model_baseline.predict(sst2_model_baseline.x_val)[1]
    logger.info("predict test with baseline")
    sst2_test_y_pred = sst2_model_baseline.predict(sst2_model_baseline.x_test)[1]
    logger.info("learn the uncertainties")
    sst2_wrapper = UncertaintyWrapper(lambda_reg, num_samples,
                                      learning_rate=learning_rate, num_hidden_units=num_units)
    sst2_wrapper.train_model(sst2_model_baseline.x_val, sst2_model_baseline.y_val, sst2_val_y_pred,
                             epochs=epochs, batch_size=batch_size)
    logger.info("predict the uncertainties")
    sst2_test_uncertainties = K.get_session().run(sst2_wrapper.predict_entropy(sst2_model_baseline.x_test, sst2_test_y_pred))
    coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    model_name = 'sst2_selectivenet'
    logger.info("train selectivenet")
    results = train_profile(model_name, coverages, model_baseline=sst2_model_baseline,
                            alpha=alpha, uncertainties=sst2_test_uncertainties,
                            epochs=seletivenet_epochs)
    save_dict("{}.json".format(model_name), results)
