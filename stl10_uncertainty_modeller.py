import tensorflow as tf

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.applications import vgg16, mobilenet_v2, resnet50
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp


import argparse
import os
import skimage
import ssl
import random

import numpy as np

from stl10_data_loader import STL10Loader
from tools import get_logger

import pickle

epsilon = 1e-10
lambda_reg = 1e-2

NUM_PREDICITION_SAMPLES = 1000


def dirichlet_aleatoric_cross_entropy(y_true, y_pred):
    """
        Loss function that applies a categorical cross entropy to the predictions
        obtained from the original model combined with a beta parameter that models
        the aleatoric noise associated with each data point. We model a Dirichlet
        pdf based on the combination of both the prediction and the beta parameter
        to sample from it and obtain the resulting output probabilities

        Parameters
        ----------
        y_true: `np.array`
            output of the model formed by the concatenation of the original prediction
            in the first num_classes positions, and a beta scalar in the last one
        y:  `np.array`
            the labels in one hot encoding format

        Returns
        -------
        an array with the associated cross-entropy for each element of the batch

    """
    # original probability distribution of the prediction among the classes
    mu_probs = y_pred[:, :num_classes]
    # beta parameter for each prediction
    beta = y_pred[:, num_classes:]
    beta = tf.broadcast_to(beta, (lambda shape: (shape[0], shape[1]))(tf.shape(mu_probs)))
    # alpha parameter based on the prediction scaled by the beta factor
    alpha = mu_probs * beta
    # defition of the Dir pdf
    dirichlet = tfp.distributions.Dirichlet(alpha)
    # sampling from the Dir to obtain different y_hats for the prediction
    z = dirichlet.sample(sample_shape=NUM_PREDICITION_SAMPLES)
    # MC
    e_probs = tf.reduce_mean(z, axis=0)
    # log of the resulting probabilities for the cross entropy formulation
    log_probs = tf.log(e_probs + epsilon)
    # cross entropy
    cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
    # we return the cross entropy plus a regularization term to prevent the beta
    # to grow forever
    return cross_entropy + lambda_reg * tf.reduce_sum(beta, axis=-1)


# metric that outputs the max/min value for the sigma logits
def max_beta(y_true, y_pred):
    beta = y_pred[:, num_classes:]
    return tf.reduce_max(beta)


def min_beta(y_true, y_pred):
    beta = y_pred[:, num_classes:]
    return tf.reduce_min(beta)


def create_uncertainty_model(learning_rate=1e-3, num_hidden_units=20):
    mu_input = Input(shape=(num_classes,))
    base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                          input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    # base_model = mobilenet_v2.MobileNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling='avg', classes=num_classes)
    # base_model = vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling='avg', classes=num_classes)
    # base_model = resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(32, 32, 3), pooling='avg', classes=num_classes)

    beta = base_model.output
    beta = Dense(num_hidden_units, activation='relu')(beta)
    beta = Dense(num_hidden_units, activation='relu')(beta)
    beta = Dense(num_hidden_units, activation='relu')(beta)
    # beta = Dense(num_hidden_units,activation='relu')(beta)
    beta = Dense(1, activation='softplus')(beta)
    output = concatenate([mu_input, beta])

    model = Model(inputs=[mu_input, base_model.input], outputs=output)
    model.compile(loss=dirichlet_aleatoric_cross_entropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[max_beta, min_beta]
                  )
    return model


def predict_cross_entropy(y_true, y_pred):
    prediction = tf.argmax(y_pred, axis=-1)
    probs = y_pred
    log_probs = tf.log(probs+epsilon)
    mu_entropy = -(tf.reduce_sum(probs * log_probs, axis=-1))
    cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
    return cross_entropy, mu_entropy, prediction


def voting(y_pred):
    mu_probs = y_pred[:,:num_classes]
    beta = y_pred[:,num_classes:]
    beta = tf.broadcast_to(beta,(lambda shape: (shape[0], shape[1]))(tf.shape(mu_probs)))
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICITION_SAMPLES)
    z = tf.reshape(z, (lambda shape: (NUM_PREDICITION_SAMPLES, shape[0], shape[1]))(tf.shape(mu_probs)))
    sampled_output = tf.argmax(z, axis=-1, output_type=tf.int32)
    sampled_output = tf.reshape(sampled_output,
                                (lambda shape: (NUM_PREDICITION_SAMPLES, -1))(tf.shape(sampled_output)))
    sampled_output = tf.one_hot(sampled_output, axis=-1, depth=2)
    sampled_output = tf.reduce_sum(sampled_output, axis=0)
    winner_classes = tf.argmax(sampled_output, axis=1)
    winner_classes = tf.one_hot(winner_classes, axis=-1, depth=2)
    sampled_output = tf.reduce_sum(sampled_output * winner_classes, axis=-1)
    return 1-sampled_output/NUM_PREDICITION_SAMPLES


def predict_dirichlet_entropy_gal(y_pred):
    mu_probs = y_pred[:,:num_classes]
    beta = y_pred[:,num_classes:]
    beta = tf.broadcast_to(beta,(lambda shape: (shape[0], shape[1]))(tf.shape(mu_probs)))
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICITION_SAMPLES)
    e_probs = tf.reduce_mean(z, axis=0)
    log_probs = tf.log(e_probs+epsilon)
    entropy = tf.reduce_sum(-e_probs*log_probs, axis=-1)
    return entropy


def get_rejection_measures(prediction, true_label, rejection_heuristic, rejection_point):
    assert len(prediction) == len(true_label) == len(rejection_heuristic)
    num_total_points = len(prediction)  # n
    num_non_rejected_points = rejection_point  # N
    num_rejected_points = num_total_points - rejection_point  # R
    rejection_percentage = (num_rejected_points / num_total_points) * 100
    accurately_classified = (np.sum(prediction[rejection_heuristic] ==
                                    true_label[rejection_heuristic]))  # a

    accurately_classified_non_rejected = (np.sum(prediction[rejection_heuristic][:rejection_point]
                                                 == true_label[rejection_heuristic][:rejection_point]))  # a_N
    accurately_classified_rejected = (np.sum(prediction[rejection_heuristic][rejection_point:]
                                             == true_label[rejection_heuristic][rejection_point:]))  # a_R
    non_rejected_accuracy = accurately_classified_non_rejected / num_non_rejected_points  # A = ||a_N|| / ||N||
    classification_quality = (
                (accurately_classified_non_rejected + (num_rejected_points - accurately_classified_rejected)) /
                num_total_points)  # Q = ||a_N||+||1-a_R|| / n
    rejection_quality = (((num_rejected_points - accurately_classified_rejected) / accurately_classified_rejected)
                         / ((
                                        num_total_points - accurately_classified) / accurately_classified))  # phi = ||1-a_R||/||a_r / ||1-a||/||a||
    return non_rejected_accuracy, classification_quality, rejection_quality, rejection_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_units', type=int, default=40,
                        help='num units')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='dir to save the trained model')
    parser.add_argument('--output_file', type=str, default='output.pkl',
                        help='file to dump the generated data')
    parser.add_argument('--input_dir', type=str, default='./',
                        help='dir to load the trained model')
    parser.add_argument('--mode', type=str, default='categorical',
                        help='prediction mode: categorical|probabilities')
    parser.add_argument('--label_mapping_file', type=str, default='imagenet_stl10_mapping.pkl',
                        help='label mapping file')

    args = parser.parse_args()
    input_dir = args.input_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_units = args.num_units
    epochs = args.epochs
    if args.mode == 'categorical':
        train_file = 'train_preds.pkl'
        test_file = 'test_preds.pkl'
    elif args.mode == 'probabilities':
        train_file = 'train_prob_preds.pkl'
        test_file = 'test_prob_preds.pkl'
    output_file = os.path.sep.join([input_dir, args.output_file])
    logger = get_logger()
    ssl._create_default_https_context = ssl._create_unverified_context

    new_shape = (224, 224, 3)
    num_classes = 10

    logger.info('Load mapping file')
    with open(args.label_mapping_file, 'rb') as file:
        imagenet_stl10_mapping = pickle.load(file)

    logger.info('Load  target dataset')
    with open(os.path.sep.join([input_dir, train_file]), 'rb') as file:
        mu_predictions = pickle.load(file)
        if args.mode == 'categorical':
            pred_y = [
                imagenet_stl10_mapping[label[0][1]] if imagenet_stl10_mapping[label[0][1]] is not None else random.randint(
                    0, 9) for label in decode_predictions(mu_predictions, top=1)]
            mu_predictions = to_categorical(pred_y, num_classes)
    with open(os.path.sep.join([input_dir, test_file]), 'rb') as file:
        test_mu_predictions = pickle.load(file)
        if args.mode == 'categorical':
            test_pred_y = [
                imagenet_stl10_mapping[label[0][1]] if imagenet_stl10_mapping[label[0][1]] is not None else random.randint(
                    0, 9) for label in decode_predictions(test_mu_predictions, top=1)]
            test_mu_predictions = to_categorical(test_pred_y, num_classes)
    logger.error("Loading STL-10")
    stl10_data_loader = STL10Loader(num_classes)
    (stl10_x_train, stl10_y_train, stl10_y_train_cat), (stl10_x_test, stl10_y_test, stl10_y_test_cat) = stl10_data_loader.load_raw_dataset()

    logger.error("Resize training images")
    stl10_x_train_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_train])
    logger.error("Resize test images")
    stl10_x_test_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_test])

    logger.info("Create uncertainty model")
    unc_model = create_uncertainty_model(learning_rate=learning_rate, num_hidden_units=num_units)
    logger.info("train uncertainty")
    training_history = unc_model.fit([mu_predictions, stl10_x_train_resized],
                                     stl10_y_train_cat,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_split=0.1)
    logger.info("Save the model")
    tf.keras.models.save_model(
        unc_model,
        output_file,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
    logger.info("Done")
