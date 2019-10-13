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
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tools import get_logger
import pickle
import random
import skimage
from stl10_data_loader import STL10Loader

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
    parser.add_argument('--input_dir', type=str, default='/data/data2/jmena/STL-10',
                        help='dir holding the data')
    parser.add_argument('--mode', type=str, default='categorical',
                        help='prediction mode: categorical|probabilities|prob_cat')
    parser.add_argument('--output_file', type=str, default='stl10_output',
                        help='file to dump the generated data')
    parser.add_argument('--label_mapping_file', type=str, default='imagenet_stl10_mapping.pkl',
                        help='label mapping file')

    logger = get_logger()
    new_shape = (224, 224, 3)
    num_classes = 10
    args = parser.parse_args()
    input_dir = args.input_dir
    if args.mode == 'categorical':
        test_file = 'test_preds.pkl'
    elif args.mode == 'probabilities':
        test_file = 'test_prob_preds.pkl'
    elif args.mode == 'prob_cat':
        test_file = 'test_prob_cat_preds.pkl'
    output_file = os.path.sep.join([input_dir, args.output_file])

    logger.info('Load mapping file')
    with open(args.label_mapping_file, 'rb') as file:
        imagenet_stl10_mapping = pickle.load(file)

    logger.error('Load pretrained data')
    logger.info('Load  test dataset')
    with open(os.path.sep.join([input_dir, test_file]), 'rb') as file:
        test_mu_predictions = pickle.load(file)
        if args.mode == 'categorical':
            test_pred_y = [
                imagenet_stl10_mapping[label[0][1]] if imagenet_stl10_mapping[label[0][1]] is not None else random.randint(
                    0, 9) for label in decode_predictions(test_mu_predictions, top=1)]
            test_mu_predictions = to_categorical(test_pred_y, num_classes)

    logger.error("Loading STL-10")
    stl10_data_loader = STL10Loader(num_classes)
    (stl10_x_train, stl10_y_train, stl10_y_train_cat), (
    stl10_x_test, stl10_y_test, stl10_y_test_cat) = stl10_data_loader.load_raw_dataset()

    logger.error("Resize test images")
    stl10_x_test_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_test])
    logger.error('Load pretrained model')
    unc_model = tf.keras.models.load_model(os.path.sep.join([input_dir, args.trained_model]),
                                           custom_objects={
                                               'dirichlet_aleatoric_cross_entropy': dirichlet_aleatoric_cross_entropy,
                                               'max_beta': max_beta,
                                               'min_beta': min_beta
                                           })

    predictions = unc_model.predict([test_mu_predictions, stl10_x_test_resized])

    logger.error("Compute uncertainty metrics")
    sess = K.get_session()
    logger.error("Compute mu pred. entropy")
    error, mu_entropy, pred_y = sess.run(predict_cross_entropy(stl10_y_test_cat, test_mu_predictions))
    logger.error("Compute variation ratios")
    voted_pred = K.get_session().run(voting(predictions))
    logger.error("Compute beta pred. entropy")
    sampling_entropy_gal = sess.run(predict_dirichlet_entropy_gal(predictions))
    logger.error("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(pred_y, stl10_y_test, np.argsort(sampling_entropy_gal),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(pred_y, stl10_y_test, np.argsort(mu_entropy), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(pred_y, stl10_y_test, np.argsort(voted_pred), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    logger.error("Export results")
    with open(output_file, 'wb') as file:
        pickle.dump((mu_entropy, error, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     stl10_y_test), file)
    logger.error("Done")
