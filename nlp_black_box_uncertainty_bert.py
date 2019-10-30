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
from bert.tokenization import FullTokenizer
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
from tqdm import tqdm

from tools import process_text, MeanEmbeddingVectorizer, process_label
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


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

def load_data(training_file, val_file, test_file):
    train_df = pd.read_csv(training_file, sep='\t', names=['sentence','label'])
    train_df = train_df.dropna()
    val_df = pd.read_csv(val_file, sep='\t', names=['sentence','label'])
    val_df = val_df.dropna()
    test_df = pd.read_csv(test_file, sep='\t', names=['sentence','label'])
    test_df = test_df.dropna()
    return train_df, val_df, test_df


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
    z = dirichlet.sample(sample_shape=NUM_TRAINING_SAMPLES)
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


def create_model(learning_rate=1e-3, num_hidden_units=20):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=0, trainable=False, pooling="first")(bert_inputs)
    probs_mu = Input(shape=(NUM_CLASSES,))
    logits_sigma = Dense(num_hidden_units, activation='relu')(bert_output)
    logits_sigma = Dense(num_hidden_units,activation='relu')(logits_sigma)
    logits_sigma = Dense(num_hidden_units,activation='relu')(logits_sigma)
    logits_sigma = Dense(num_hidden_units,activation='relu')(logits_sigma)
    logits_sigma = Dense(1, activation='softplus')(logits_sigma)
    output = concatenate([probs_mu, logits_sigma])

    model = Model(inputs=[in_id, in_mask, in_segment, probs_mu], outputs=output)
    model.compile(loss=dirichlet_aleatoric_cross_entropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[mu_accuracy])
    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--training_file', type=str, default='training.tsv',
                        help='file with the training set')
    parser.add_argument('--val_file', type=str, default='val.tsv',
                        help='file with the validation set')
    parser.add_argument('--test_file', type=str, default='val.tsv',
                        help='file with the test set')
    parser.add_argument('--training_preds', type=str, default='training_preds.p',
                        help='file with the training set')
    parser.add_argument('--val_preds', type=str, default='val_preds.p',
                        help='file with the validation set')
    parser.add_argument('--test_preds', type=str, default='test_preds.p',
                        help='file with the test set')
    parser.add_argument('--output_model_file', type=str, default='yelp2013_unc_model_bert',
                        help='file to dump the generated model')
    parser.add_argument('--output_history_file', type=str, default='yelp2013_unc_history_elmo',
                        help='file to dump the generated data')
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--num_hidden_units', type=int, default=40,
                        help='num hidden units')
    parser.add_argument('--bert_path', type=str, default='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                        help='elmo model')

    args = parser.parse_args()
    logger = get_logger()
    logger.info("Load data")
    train_df, val_df, test_df = load_data(args.training_file, args.val_file, args.test_file)
    logger.info("Preprocess train data")
    train_df = preprocess(train_df)
    logger.info("Preprocess val data")
    val_df = preprocess(val_df)
    logger.info("Preprocess test data")
    test_df = preprocess(test_df)

    NUM_TRAINING_SAMPLES = 1000
    NUM_PREDICTION_SAMPLES = 1000
    NUM_CLASSES = 2
    EPSILON = 1e-10
    LAMBDA_REG = 1e-2
    # Params for bert model and tokenization
    bert_path = args.bert_path

    # Initialize session
    sess = tf.Session()

    logger.info("Prepare features")
    y_train = train_df['sentiment'].values  # target
    y_train = np.eye(NUM_CLASSES)[y_train.astype(dtype=np.int32)]
    y_val = val_df['sentiment'].values  # target
    y_val = np.eye(NUM_CLASSES)[y_val.astype(dtype=np.int32)]
    y_test = test_df['sentiment'].values  # target
    y_test = np.eye(NUM_CLASSES)[y_test.astype(dtype=np.int32)]
    max_seq_length = 120
    # Create datasets (Only take up to 120 words for memory)
    train_text = train_df['clean_text'].tolist()
    train_text = [' '.join(t[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]

    val_text = val_df['clean_text'].tolist()
    val_text = [' '.join(t[0:max_seq_length]) for t in val_text]
    val_text = np.array(val_text, dtype=object)[:, np.newaxis]

    test_text = test_df['clean_text'].tolist()
    test_text = [' '.join(t[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, y_train)
    val_examples = convert_text_to_examples(val_text, y_val)
    test_examples = convert_text_to_examples(test_text, y_test)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels
     ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
    (val_input_ids, val_input_masks, val_segment_ids, val_labels
     ) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels
     ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

    logger.info("Save the predictions")
    with open(args.training_preds, 'rb') as file:
        train_preds = pickle.load(file)
    with open(args.val_preds, 'rb') as file:
        val_preds = pickle.load(file)
    with open(args.test_preds, 'rb') as file:
        test_preds = pickle.load(file)

    logger.info("Create model")
    model = create_model(learning_rate=args.learning_rate, num_hidden_units=args.num_hidden_units)
    # Instantiate variables
    initialize_vars(sess)
    logger.info("Fit the model")
    training_history = model.fit([train_input_ids,
                                  train_input_masks,
                                  train_segment_ids, train_preds],
                                 y_train,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 shuffle=True,
                                 verbose=1,
                                 validation_data=([val_input_ids,
                                                   val_input_masks,
                                                   val_segment_ids, val_preds], y_val))
    logger.info("Save model")
    tf.keras.models.save_model(
        model,
        args.output_model_file,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
    logger.info("Save the history")
    with open(args.output_history_file, 'wb') as file:
        pickle.dump(training_history.history, file)
    logger.info("Done")
