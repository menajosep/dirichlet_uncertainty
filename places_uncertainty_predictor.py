import os

import tensorflow as tf
import warnings

from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.utils import get_source_inputs, to_categorical
from tensorflow.python.keras.utils.data_utils import get_file, Sequence
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.applications import vgg16, mobilenet_v2, resnet50

from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp
from cv2 import resize
from tqdm import tqdm

import argparse

from places_uncertainty_modeller import dirichlet_aleatoric_cross_entropy, max_beta, min_beta

num_classes = 365
import numpy as np
import pandas as pd
from tools import get_logger
import pickle

epsilon = 1e-10
lambda_reg = 1e-2

NUM_PREDICITION_SAMPLES = 900


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
    parser.add_argument('--trained_model', type=str, default='unc_model',
                        help='dir holding the data')
    parser.add_argument('--input_dir', type=str, default='/Users/jose.mena/dev/personal/data/Places365-Challenge',
                        help='dir holding the data')
    parser.add_argument('--output_file', type=str, default='places365',
                        help='file to dump the generated data')

    logger = get_logger()
    args = parser.parse_args()
    input_dir = args.input_dir
    data_dir = os.path.sep.join([input_dir, 'data_256'])
    output_file = os.path.sep.join([input_dir, args.output_file])

    logger.error('Load pretrained data')
    test_file = os.path.sep.join([input_dir, 'test_data.pkl'])
    with open(test_file, 'rb') as file:
        (test_file_names, test_labels, test_predictions) = pickle.load(file, encoding='latin1')
        test_predictions = np.array(test_predictions)
    test_images_idle = os.path.sep.join([input_dir, 'places365_test_images.pkl'])
    with open(test_images_idle, 'rb') as file:
        test_images = np.array(pickle.load(file, encoding='latin1'))
    logger.error('Load pretrained model')
    unc_model = tf.keras.models.load_model(os.path.sep.join([input_dir, args.trained_model]),
                                           custom_objects={
                                               'dirichlet_aleatoric_cross_entropy': dirichlet_aleatoric_cross_entropy,
                                               'max_beta': max_beta,
                                               'min_beta': min_beta
                                           })

    predictions = unc_model.predict([test_predictions, test_images])

    logger.error("Compute uncertainty metrics")
    sess = K.get_session()
    logger.error("Compute mu pred. entropy")
    error, mu_entropy, pred_y = sess.run(predict_cross_entropy(to_categorical(test_labels, num_classes), test_predictions))
    logger.error("Compute variation ratios")
    voted_pred = K.get_session().run(voting(predictions))
    logger.error("Compute beta pred. entropy")
    sampling_entropy_gal = sess.run(predict_dirichlet_entropy_gal(predictions))
    logger.error("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(pred_y, test_labels, np.argsort(sampling_entropy_gal),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(pred_y, test_labels, np.argsort(mu_entropy), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(pred_y, test_labels, np.argsort(voted_pred), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    logger.error("Export results")
    with open(output_file, 'wb') as file:
        pickle.dump((mu_entropy, error, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     test_images, test_labels), file)
    logger.error("Done")
