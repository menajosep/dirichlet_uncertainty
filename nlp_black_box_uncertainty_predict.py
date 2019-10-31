import argparse
import pickle

import gensim

import pandas as pd
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import Dense, Dropout, Input, concatenate, Activation
from tensorflow.python.keras.layers import BatchNormalization, Lambda
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.constraints import MaxNorm
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub

from tools import process_text, MeanEmbeddingVectorizer, process_label, create_tokenizer_from_hub_module, \
    convert_text_to_examples, convert_examples_to_features
from tools import get_logger


# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

def load_data(test_file):
    test_df = pd.read_csv(test_file, sep='\t', names=['sentence','label'])
    test_df = test_df.dropna()
    return test_df


def preprocess(df):
    df['clean_text'] = df['sentence'].apply(process_text)
    df = df[df['clean_text'].map(lambda d: len(d)) > 0]
    df['sentiment'] = df['label'].apply(process_label)
    return df


def dirichlet_aleatoric_cross_entropy(y_true, y_pred):
    # logits_mu = y_pred[:,:NUM_CLASSES]
    mu_probs = y_pred[:, :NUM_CLASSES]
    logits_sigma = y_pred[:, NUM_CLASSES:]
    beta = logits_sigma
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICTION_SAMPLES)
    e_probs = tf.reduce_mean(z, axis=0)
    log_probs = tf.log(e_probs + EPSILON)
    cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
    return cross_entropy + LAMBDA_REG * tf.reduce_sum(beta, axis=-1)


# metric that outputs the max/min value for the sigma logits
def max_sigma(y_true, y_pred):
    logits_psi = y_pred[:, NUM_CLASSES:]
    return tf.reduce_max(logits_psi)


def min_sigma(y_true, y_pred):
    logits_psi = y_pred[:, NUM_CLASSES:]
    return tf.reduce_min(logits_psi)


# metric that outputs the accuracy when only considering the logits_mu.
# this accuracy should be the same that was obtained with the fake classifier
# in its best epoch.
def mu_accuracy(y_true, y_pred):
    logits_phi = y_pred[:, :NUM_CLASSES]
    labels_phi = y_true[:, :NUM_CLASSES]
    return categorical_accuracy(labels_phi, logits_phi)


def predict_cross_entropy(y_true, y_pred):
    prediction = tf.argmax(y_pred, axis=-1)
    probs = y_pred
    log_probs = tf.log(probs+EPSILON)
    mu_entropy = -(tf.reduce_sum(probs * log_probs, axis=-1))
    cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
    return cross_entropy, mu_entropy, prediction


def voting(y_pred):
    mu_probs = y_pred[:,:NUM_CLASSES]
    beta = y_pred[:,NUM_CLASSES:]
    beta = tf.broadcast_to(beta,(lambda shape: (shape[0], shape[1]))(tf.shape(mu_probs)))
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICTION_SAMPLES)
    z = tf.reshape(z, (lambda shape: (NUM_PREDICTION_SAMPLES, shape[0], shape[1]))(tf.shape(mu_probs)))
    sampled_output = tf.argmax(z, axis=-1, output_type=tf.int32)
    sampled_output = tf.reshape(sampled_output,
                                (lambda shape: (NUM_PREDICTION_SAMPLES, -1))(tf.shape(sampled_output)))
    sampled_output = tf.one_hot(sampled_output, axis=-1, depth=2)
    sampled_output = tf.reduce_sum(sampled_output, axis=0)
    winner_classes = tf.argmax(sampled_output, axis=1)
    winner_classes = tf.one_hot(winner_classes, axis=-1, depth=2)
    sampled_output = tf.reduce_sum(sampled_output * winner_classes, axis=-1)
    return 1-sampled_output/NUM_PREDICTION_SAMPLES


def predict_dirichlet_entropy_gal(y_pred):
    mu_probs = y_pred[:,:NUM_CLASSES]
    beta = y_pred[:,NUM_CLASSES:]
    beta = tf.broadcast_to(beta,(lambda shape: (shape[0], shape[1]))(tf.shape(mu_probs)))
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_PREDICTION_SAMPLES)
    e_probs = tf.reduce_mean(z, axis=0)
    log_probs = tf.log(e_probs+EPSILON)
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
    parser.add_argument('--test_file', type=str, default='val.tsv',
                        help='file with the test set')
    parser.add_argument('--test_preds', type=str, default='test_preds.p',
                        help='file with the test set')
    parser.add_argument('--uncer_model_file', type=str, default='yelp2013_unc_model',
                        help='file to load the generated model')
    parser.add_argument('--mode', type=str, default='ELMO',
                        help='type of embedding model ELMO|BERT|W2V')
    parser.add_argument('--bert_path', type=str, default='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                        help='elmo model')
    parser.add_argument('--output_file', type=str, default='yelp2013_output',
                        help='file to dump the generated data')

    args = parser.parse_args()
    logger = get_logger()
    logger.info("Load data")
    test_df = load_data(args.test_file)
    logger.info("Preprocess test data")
    test_df = preprocess(test_df)

    NUM_PREDICTION_SAMPLES = 1000
    NUM_PREDICTION_SAMPLES = 1000
    NUM_CLASSES = 2
    EPSILON = 1e-10
    LAMBDA_REG = 1e-2

    # Initialize session
    sess = tf.Session()

    logger.info("Prepare features")
    y_test = test_df['sentiment'].values  # target
    y_test = np.eye(NUM_CLASSES)[y_test.astype(dtype=np.int32)]
    max_seq_length = 120
    # Create datasets (Only take up to 120 words for memory)
    test_text = test_df['clean_text'].tolist()
    test_text = [' '.join(t[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    logger.info("Load the predictions")
    with open(args.test_preds, 'rb') as file:
        test_preds = pickle.load(file)
    inputs = [test_text, test_preds]
    if args.mode == 'BERT':
        # Instantiate tokenizer
        tokenizer = create_tokenizer_from_hub_module(sess, bert_path=args.bert_path)

        # Convert data to InputExample format
        test_examples = convert_text_to_examples(test_text, y_test)

        # Convert to features
        (test_input_ids, test_input_masks, test_segment_ids, test_labels
         ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)
        inputs = [test_input_ids, test_input_masks, test_segment_ids, test_preds]


    logger.info("Load model")
    unc_model = tf.keras.models.load_model(args.uncer_model_file,
                                           custom_objects={
                                               'dirichlet_aleatoric_cross_entropy': dirichlet_aleatoric_cross_entropy,
                                               'max_sigma': max_sigma,
                                               'min_sigma': min_sigma,
                                               'mu_accuracy': mu_accuracy,
                                               'BertLayer': BertLayer
                                           })
    logger.info("Fit the model")
    predictions = unc_model.predict(inputs)
    logger.error("Compute uncertainty metrics")
    logger.error("Compute mu pred. entropy")
    error, mu_entropy, pred_y = sess.run(predict_cross_entropy(y_test, test_preds))
    logger.error("Compute variation ratios")
    voted_pred = K.get_session().run(voting(predictions))
    logger.error("Compute beta pred. entropy")
    sampling_entropy_gal = sess.run(predict_dirichlet_entropy_gal(predictions))
    logger.error("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(pred_y, y_test, np.argsort(sampling_entropy_gal),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(pred_y, y_test, np.argsort(mu_entropy), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(pred_y, y_test, np.argsort(voted_pred), rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    logger.error("Export results")
    with open(args.output_file, 'wb') as file:
        pickle.dump((mu_entropy, error, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     y_test), file)
    logger.error("Done")
