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
from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling

epsilon = 1e-10
lambda_reg = 1e-2

NUM_PREDICITION_SAMPLES = 900


def predict_probs(y_pred):
    logits_mu = y_pred[:,:num_classes]
    mu_probs = tf.nn.softmax(logits_mu, axis=-1)
    logits_sigma = y_pred[:,num_classes:]
    beta = logits_sigma
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICITION_SAMPLES)
    e_probs = tf.reduce_mean(z, axis=0)
    return e_probs


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

    logger.error("Compute predictions")
    sess = K.get_session()
    logger.error("Compute mu pred. entropy")
    probs = sess.run(predict_probs(predictions))
    accuracy = accuracy_score(stl10_y_test, np.argmax(predictions, axis=1))
    logger.error("Resulting accuracy: {}".format(accuracy))

    n_bins = 10
    ground_truth = stl10_y_test
    confidences = test_mu_predictions

    temperature = TemperatureScaling()
    temperature.fit(confidences, ground_truth)
    calibrated = temperature.transform(confidences)
    n_bins = 10

    ece = ECE(n_bins)
    uncalibrated_score = ece.measure(confidences, ground_truth)
    calibrated_score = ece.measure(calibrated, ground_truth)
    wrapper_score = ece.measure(predictions, ground_truth)
    logger.error("ECE scores: {}, {}, {}".format(uncalibrated_score, calibrated_score, wrapper_score))
    logger.error("Done")
