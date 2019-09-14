import tensorflow as tf

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Input, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.applications import vgg16, mobilenet_v2, resnet50

from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp


import argparse
import skimage

import numpy as np

from cifar_data_loader import Cifar10Loader
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
                                          input_shape=(32, 32, 3), pooling='avg', classes=num_classes)
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='epochs to train uncertainty model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='dir to save the trained model')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes for the trained model')
    parser.add_argument('--target', type=str, default='cifar10',
                        help='target dataset where to compute the uncertainty')
    parser.add_argument('--output_file', type=str, default='output.pkl',
                        help='file to dump the generated data')

    args = parser.parse_args()
    logger = get_logger()

    num_classes = args.num_classes
    target_dataset = args.target
    logger.info('Load pretrained model')
    trained_model = tf.keras.models.load_model(args.save_dir)
    logger.info('Load  target dataset')

    if target_dataset == 'stl10':

        stl10_data_loader = STL10Loader(num_classes)
        (stl10_x_train, stl10_y_train, stl10_y_train_cifar10, stl10_y_train_cat), (
        stl10_x_test, stl10_y_test, stl10_y_test_cifar10, stl10_y_test_cat) = stl10_data_loader.load_dataset()
        print(stl10_x_train.shape, stl10_y_train.shape, stl10_x_test.shape, stl10_y_test.shape)

        new_shape = (32, 32, 3)
        stl10_x_train_resized = np.array(
            [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_train])
        stl10_x_test_resized = np.array(
            [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_test])
        x_train = stl10_x_train_resized
        y_train = stl10_y_train_cat
        x_test = stl10_x_test_resized
        y_test = stl10_y_test_cat
    else:
        cifar10_data_loader = Cifar10Loader(args.num_classes)

        y_train, y_test, x_train, x_test = cifar10_data_loader.get_cifar10_data()

    transfer_scores = trained_model.evaluate(x_test, y_test, verbose=1)
    logger.info('Test loss: {}'.format(transfer_scores[0]))
    logger.info('Test accuracy: {}'.format(transfer_scores[1]))
    logger.info("Apply trained model to target dataset to obtain the mus")
    mu_predictions = trained_model.predict(x_train)
    test_mu_predictions = trained_model.predict(x_test)
    logger.info("Create uncertainty model")
    unc_model = create_uncertainty_model(learning_rate=1e-4, num_hidden_units=40)
    logger.info("train uncertainty")
    training_history = unc_model.fit([mu_predictions, x_train],
                                     y_train,
                                     batch_size=args.batch_size,
                                     epochs=args.epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_split=0.2)
    logger.info("Predict the uncertainty")
    predictions = unc_model.predict([test_mu_predictions, x_test])
    logger.info("Compute uncertainty metrics")
    error, mu_entropy, pred_y = K.get_session().run(predict_cross_entropy(y_test, test_mu_predictions))
    voted_pred = K.get_session().run(voting(predictions))
    sampling_entropy_gal = K.get_session().run(predict_dirichlet_entropy_gal(predictions))
    logger.info("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(sampling_entropy_gal), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(mu_entropy), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(voted_pred), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    with open(args.output_file, 'wb') as file:
        pickle.dump((mu_entropy, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     x_test, y_test), file)
    logger.info("Done")
