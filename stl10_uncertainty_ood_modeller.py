import argparse
import os
import pickle
import random
import ssl

import numpy as np
import skimage
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.applications import vgg16, mobilenet_v2, resnet50, vgg19, inception_v3
from tensorflow.python.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical

from stl10_data_loader import STL10Loader
from tools import get_logger

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
        y_true:  `np.array`
            the labels in one hot encoding format

        y_pred: `np.array`
            output of the model formed by the concatenation of the original prediction
            in the first num_classes positions, and a beta scalar in the last one


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
    cross_entropy = -(tf.reduce_sum(y_true[:, :num_classes] * log_probs, axis=-1))
    # we return the cross entropy plus a regularization term to prevent the beta
    # to grow forever
    aleatoric_loss = cross_entropy + lambda_reg * tf.reduce_sum(beta, axis=-1)
    return y_true[:, num_classes:] * aleatoric_loss + (1 - y_true[:, num_classes:]) * tf.reduce_sum(beta, axis=-1)


# metric that outputs the max/min value for the sigma logits
def max_beta(y_true, y_pred):
    beta = y_pred[:, num_classes:]
    return tf.reduce_max(beta)


def min_beta(y_true, y_pred):
    beta = y_pred[:, num_classes:]
    return tf.reduce_min(beta)


def create_uncertainty_model(learning_rate=1e-3, num_hidden_units=20, type = 'mobilenet_v2'):
    mu_input = Input(shape=(num_classes,))
    if type == 'mobilenet_v2':
        base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                          input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    elif type == 'vgg16':
        base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                 input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    elif type == 'resnet50':
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                 input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    elif type == 'vgg19':
        base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None,
                                 input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    elif type == 'inception_v3':
        base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None,
                                 input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    else:
        base_model = mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=(224, 224, 3), pooling='avg', classes=num_classes)
    base_model.trainable = False
    beta = base_model.output
    beta = Dense(num_hidden_units, activation='relu')(beta)
    beta = Dense(num_hidden_units, activation='relu')(beta)
    beta = Dense(num_hidden_units, activation='relu')(beta)
    # beta = Dense(num_hidden_units,activation='relu')(beta)
    beta = Dense(1, activation='sigmoid')(beta)
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
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_units', type=int, default=40,
                        help='num units')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--output_model_file', type=str, default='stl10_uncertainty_model',
                        help='file to dump the generated model')
    parser.add_argument('--output_history_file', type=str, default='stl10_uncertainty_history',
                        help='file to dump the generated data')
    parser.add_argument('--input_dir', type=str, default='./',
                        help='dir to load the trained model')
    parser.add_argument('--label_mapping_file', type=str, default='imagenet_stl10_mapping.pkl',
                        help='label mapping file')
    parser.add_argument('--inverse_label_mapping_file', type=str, default='stl10_imagenet_mapping.pkl',
                        help='label mapping file')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                        help='Lambda parameter for regularization of beta values')
    parser.add_argument('--model_type', type=str, default='mobilenet_v2',
                        help='type of base model to learn the betas mobilenet_v2|vgg16|vgg19|resnet50|inception_v3')

    args = parser.parse_args()
    input_dir = args.input_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_units = args.num_units
    epochs = args.epochs
    train_file = 'train_prob_preds.pkl'
    output_model_file = os.path.sep.join([input_dir, args.output_model_file])
    output_history_file = os.path.sep.join([input_dir, args.output_history_file])
    lambda_reg = args.lambda_reg
    logger = get_logger()
    ssl._create_default_https_context = ssl._create_unverified_context

    new_shape = (224, 224, 3)
    num_classes = 10

    logger.info('Load mapping file')
    with open(args.label_mapping_file, 'rb') as file:
        imagenet_stl10_mapping = pickle.load(file)

    with open(args.inverse_label_mapping_file, 'rb') as file:
        stl10_imagenet_mapping = pickle.load(file)

    logger.info('Load  target dataset')
    with open(os.path.sep.join([input_dir, train_file]), 'rb') as file:
        mu_predictions = pickle.load(file)
    logger.error("Loading STL-10")
    stl10_data_loader = STL10Loader(num_classes)
    (stl10_x_train, stl10_y_train, stl10_y_train_cat), (_, _, _) = stl10_data_loader.load_raw_dataset()
    stl10_unlabeled_train, _ = stl10_data_loader.load_raw_ood_dataset(num_training=1000, num_test=1000)

    logger.error("Resize training images")
    stl10_x_train_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_x_train])
    stl10_unlabeled_train_resized = np.array(
        [skimage.transform.resize(image, new_shape, anti_aliasing=True) for image in stl10_unlabeled_train])

    stl10_train_X_ood = np.concatenate((stl10_x_train_resized, stl10_unlabeled_train_resized))
    ood_label = 1/num_classes
    ood_labels = [ood_label]*num_classes + [0]
    stl10_train_y_ood = np.concatenate(
        (np.insert(stl10_y_train_cat, num_classes, 1, axis=-1), np.array([ood_labels] * stl10_unlabeled_train_resized.shape[0])))

    logger.error("Load model")
    model = mobilenet_v2.MobileNetV2(weights='imagenet')
    logger.error("Predict training")
    train_preds = model.predict(stl10_train_X_ood)
    train_prob_preds = train_preds[:, stl10_imagenet_mapping[0]].sum(axis=1) / len(stl10_imagenet_mapping[0])
    for i in range(1, 10):
        train_prob_preds = np.hstack(
            (train_prob_preds, train_preds[:, stl10_imagenet_mapping[i]].sum(axis=1) / len(stl10_imagenet_mapping[0])))
    train_prob_preds = train_prob_preds.reshape((num_classes, train_preds.shape[0])).T
    fake_y_pred_train_ood = train_prob_preds / train_prob_preds.sum(axis=1, keepdims=1)

    train_indexes = np.random.permutation(stl10_train_X_ood.shape[0])
    stl10_train_X_ood = stl10_train_X_ood[train_indexes]
    stl10_train_y_ood = stl10_train_y_ood[train_indexes]
    fake_y_pred_train_ood = fake_y_pred_train_ood[train_indexes]
    logger.info("Create uncertainty model")
    unc_model = create_uncertainty_model(learning_rate=learning_rate, num_hidden_units=num_units, type=args.model_type)
    logger.info("train uncertainty")
    training_history = unc_model.fit([fake_y_pred_train_ood, stl10_train_X_ood],
                                     stl10_train_y_ood,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_split=0.1)

    logger.info("Save the training history")
    with open(output_history_file, 'wb') as file:
        pickle.dump(training_history.history, file)
    logger.info("Save the model")
    tf.keras.models.save_model(
        unc_model,
        output_model_file,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
    logger.info("Done")
