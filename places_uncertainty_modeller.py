import os

import tensorflow as tf
import warnings

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


import argparse
num_classes = 365
import numpy as np
import pandas as pd
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


def create_uncertainty_model(trained_model, learning_rate=1e-3, num_hidden_units=20):
    trained_model.trainable = False
    mu_input = trained_model.output
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

    model = Model(inputs=[trained_model.input, base_model.input], outputs=output)
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

WEIGHTS_PATH = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/GKalliatakis/Keras-VGG16-places365/releases/download/v1.0/vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16_Places365(include_top=True, weights='places',
                    input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=365):
    """Instantiates the VGG16-places365 architecture.

    Optionally loads weights pre-trained
    on Places. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
                 'places' (pre-training on Places),
                 or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`, or invalid input shape
        """
    if not (weights in {'places', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `places` '
                         '(pre-training on Places), '
                         'or the path to the weights file to be loaded.')

    if weights == 'places' and include_top and classes != 365:
        raise ValueError('If using `weights` as places with `include_top`'
                         ' as true, `classes` should be 365')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)

        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)

        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')

    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    elif weights is not None:
        model.load_weights(weights)

    return model


def generator_duplicate_img(generator, generator2):
    while True:
        batch1 = generator.next
        batch2 = generator2.next
        yield [batch1[0], batch2[0]], batch1[1]


class MultipleInputGenerator(Sequence):
    """Wrapper of 2 ImageDataGenerator"""

    def __init__(self, data_df, train_data_dir, batch_size):
        # Keras generator
        self.generator = ImageDataGenerator()

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow_from_dataframe(dataframe=data_df, directory=train_data_dir, x_col="file_name",
                                                      y_col="class",
                                                      class_mode="categorical", target_size=(224, 224),
                                                      batch_size=batch_size)
        self.genX2 = self.generator.flow_from_dataframe(dataframe=data_df, directory=train_data_dir, x_col="file_name",
                                                      y_col="class",
                                                      class_mode="categorical", target_size=(224, 224),
                                                      batch_size=batch_size)

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)

        X_batch = [X1_batch, X2_batch]

        return X_batch, Y_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--input_dir', type=str, default='/Users/jose.mena/dev/personal/data/Places365-Standard',
                        help='target dataset where to compute the uncertainty')
    parser.add_argument('--output_file', type=str, default='places365_output.pkl',
                        help='file to dump the generated data')

    args = parser.parse_args()
    input_dir = args.input_dir
    train_data_dir = os.path.sep.join([input_dir, 'data_256'])
    batch_size = args.batch_size

    logger = get_logger()

    logger.info('Load pretrained model')
    trained_model = VGG16_Places365(weights='places')
    trained_model.trainable = False
    logger.info('Load  target dataset')
    data_file = 'places365_train_standard.txt'
    #data_file = 'places365_train_challenge.txt'
    data_file = os.path.sep.join([input_dir, data_file])
    test_data_file = 'places365_test.txt'
    test_data_file = os.path.sep.join([input_dir, test_data_file])

    data_df = pd.read_csv(data_file, sep=' ', names=['id', 'label'])
    data_df['class'] = data_df['label'].astype(str)
    data_df['num_index'] = data_df.id.str[-12:-4].astype(int)
    data_df['file_name'] = data_df['id'].map(
        lambda fname: train_data_dir+'/'+fname
    )
    #data_df = data_df[data_df.num_index > 5000]
    msk = np.random.rand(len(data_df)) < 0.9
    train_test_df = data_df[msk]
    val_df = data_df[~msk]
    test_msk = np.random.rand(len(train_test_df)) < 0.9
    train_df = train_test_df[test_msk]
    test_df = train_test_df[~test_msk]

    train_generator = MultipleInputGenerator(train_df, train_data_dir, batch_size)
    val_generator = MultipleInputGenerator(val_df, train_data_dir, batch_size)
    test_generator = MultipleInputGenerator(test_df, train_data_dir, batch_size)

    trained_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['acc'])

    transfer_scores = trained_model.evaluate_generator(val_generator, steps=len(val_df)//batch_size, verbose=1)
    logger.info('Test loss: {}'.format(transfer_scores[0]))
    logger.info('Test accuracy: {}'.format(transfer_scores[1]))
    logger.info("Create uncertainty model")
    unc_model = create_uncertainty_model(trained_model, learning_rate=1e-4, num_hidden_units=40)
    logger.info("train uncertainty")
    training_history = unc_model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_df) // batch_size,
                                     epochs=args.epochs,
                                     shuffle=True,
                                     verbose=1,
                                     validation_data=val_generator,
                                     validation_steps=len(val_df) // batch_size)
    logger.info("Predict the uncertainty")
    test_mu_predictions = trained_model.predict_generator(test_generator, steps=len(test_df) // batch_size)
    y_test = to_categorical(test_df.label.values, 365)
    predictions = unc_model.predict_generator(test_generator, steps=len(test_df) // batch_size)

    logger.info("Compute uncertainty metrics")
    error, mu_entropy, pred_y = K.get_session().run(predict_cross_entropy(y_test, test_mu_predictions))
    voted_pred = K.get_session().run(voting(predictions))
    sampling_entropy_gal = K.get_session().run(predict_dirichlet_entropy_gal(predictions))
    logger.info("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(sampling_entropy_gal),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(mu_entropy), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(pred_y, np.argmax(y_test, axis=1), np.argsort(voted_pred), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    logger.info("Export results")
    with open(args.output_file, 'wb') as file:
        pickle.dump((mu_entropy, error, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     y_test), file)
    logger.info("Done")
