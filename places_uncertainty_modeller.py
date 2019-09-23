import argparse
import os

import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image
from cv2 import resize
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tqdm import tqdm

num_classes = 365
import numpy as np
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--num_units', type=int, default=20,
                        help='num units')
    parser.add_argument('--input_dir', type=str, default='/Users/jose.mena/dev/personal/data/Places365-Challenge',
                        help='dir holding the data')
    parser.add_argument('--output_file', type=str, default='places365',
                        help='file to dump the generated data')

    logger = get_logger()
    args = parser.parse_args()
    input_dir = args.input_dir
    data_dir = os.path.sep.join([input_dir, 'data_256'])
    output_file = os.path.sep.join([input_dir, args.output_file])
    logger.info('Load pretrained data')
    train_file = os.path.sep.join([input_dir, 'train_data.pkl'])
    test_file = os.path.sep.join([input_dir, 'test_data.pkl'])
    with open(train_file, 'rb') as file:
        (train_file_names, train_labels, train_predictions) = pickle.load(file, encoding='latin1')
        train_predictions = np.array(train_predictions)
    with open(test_file, 'rb') as file:
        (test_file_names, test_labels, test_predictions) = pickle.load(file, encoding='latin1')
        test_predictions = np.array(test_predictions)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_units = args.num_units
    epochs = args.epochs
    num_classes = np.max(train_labels) + 1

    test_images = []
    for file_name in tqdm(test_file_names):
        with open(os.path.sep.join([data_dir, file_name]), 'rb') as image_file:
            image = Image.open(image_file)
            image = np.array(image, dtype=np.uint8)
            image = resize(image, (224, 224))
            test_images.append(image)
    test_images = np.array(test_images)

    train_images = []
    for file_name in tqdm(train_file_names):
        with open(os.path.sep.join([data_dir, file_name]), 'rb') as image_file:
            image = Image.open(image_file)
            image = np.array(image, dtype=np.uint8)
            image = resize(image, (224, 224))
            train_images.append(image)
    train_images = np.array(train_images)




    logger.info("Create uncertainty model")
    unc_model = create_uncertainty_model(learning_rate=learning_rate, num_hidden_units=num_units)
    logger.info("train uncertainty")
    training_history = unc_model.fit([train_predictions, train_images],
                                     to_categorical(train_labels, num_classes),
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_split=0.2)
    logger.info("Save the model")
    tf.keras.models.save_model(
        unc_model,
        output_file,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )

    logger.info("Done")
