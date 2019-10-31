import argparse
import gensim
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_probability as tfp
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Input, concatenate
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from tools import get_logger, MeanEmbeddingVectorizer
from tools import process_text, process_label, predict_cross_entropy, voting, \
    predict_dirichlet_entropy_gal, get_rejection_measures



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
    logits_beta = y_pred[:, NUM_CLASSES:]
    beta = logits_beta
    alpha = mu_probs * beta
    dirichlet = tfp.distributions.Dirichlet(alpha)
    z = dirichlet.sample(sample_shape=NUM_TRAINING_SAMPLES)
    e_probs = tf.reduce_mean(z, axis=0)
    log_probs = tf.log(e_probs + EPSILON)
    cross_entropy = -(tf.reduce_sum(y_true * log_probs, axis=-1))
    return cross_entropy + LAMBDA_REG * tf.reduce_sum(beta, axis=-1)


# metric that outputs the max/min value for the beta logits
def max_beta(y_true, y_pred):
    logits_psi = y_pred[:, NUM_CLASSES:]
    return tf.reduce_max(logits_psi)


def min_beta(y_true, y_pred):
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
    model_input = Input(shape=(OUTPUT_LAYER_SIZE,))
    probs_mu = Input(shape=(NUM_CLASSES,))
    logits_beta = Dense(num_hidden_units, activation='relu')(model_input)
    logits_beta = Dense(num_hidden_units,activation='relu')(logits_beta)
    logits_beta = Dense(num_hidden_units,activation='relu')(logits_beta)
    logits_beta = Dense(num_hidden_units,activation='relu')(logits_beta)
    logits_beta = Dense(1, activation='softplus')(logits_beta)
    output = concatenate([probs_mu, logits_beta])

    model = Model(inputs=[model_input, probs_mu], outputs=output)
    model.compile(loss=dirichlet_aleatoric_cross_entropy,
                  optimizer=Adam(lr=learning_rate),
                  metrics=[mu_accuracy, max_beta, min_beta])
    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def load_w2v_model(google_w2v_file):
    google_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(google_w2v_file, binary=True)
    return dict(zip(google_model.wv.index2word, google_model.wv.syn0))


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
    parser.add_argument('--output_model_file', type=str, default='yelp2013_unc_model',
                        help='file to dump the generated model')
    parser.add_argument('--output_history_file', type=str, default='yelp2013_unc_history_w2v',
                        help='file to dump the generated data')
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--num_hidden_units', type=int, default=40,
                        help='num hidden units')
    parser.add_argument('--google_w2v_file', type=str, default='GoogleNews-vectors-negative300.bin.gz',
                        help='google_w2v_file')
    parser.add_argument('--output_file', type=str, default='yelp2013_output',
                        help='file to dump the generated data')

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

    logger.info("Prepare features")
    w2v = load_w2v_model(args.google_w2v_file)
    w2v_vectorizer = MeanEmbeddingVectorizer(w2v)
    X_train = w2v_vectorizer.transform(train_df['clean_text'])  # features
    y_train = train_df['sentiment'].values  # target
    y_train = np.eye(NUM_CLASSES)[y_train.astype(dtype=np.int32)]
    X_val = w2v_vectorizer.transform(val_df['clean_text'])  # features
    y_val = val_df['sentiment'].values  # target
    y_val = np.eye(NUM_CLASSES)[y_val.astype(dtype=np.int32)]
    X_test = w2v_vectorizer.transform(test_df['clean_text'])  # features
    y_test = test_df['sentiment'].values  # target
    y_test = np.eye(NUM_CLASSES)[y_test.astype(dtype=np.int32)]

    OUTPUT_LAYER_SIZE = X_train.shape[1]

    logger.info("Save the predictions")
    with open(args.training_preds, 'rb') as file:
        train_preds = pickle.load(file)
    with open(args.val_preds, 'rb') as file:
        val_preds = pickle.load(file)
    with open(args.test_preds, 'rb') as file:
        test_preds = pickle.load(file)

    logger.info("Create model")
    # Initialize session
    sess = tf.Session()
    model = create_model(learning_rate=args.learning_rate, num_hidden_units=args.num_hidden_units)
    # Instantiate variables
    initialize_vars(sess)
    logger.info("Fit the model")
    training_history = model.fit([X_train, train_preds],
                                 y_train,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs,
                                 shuffle=True,
                                 verbose=1,
                                 validation_data=([X_val, val_preds], y_val))
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

    test_inputs = [X_test, test_preds]
    logger.info("Fit the model")
    predictions = model.predict(test_inputs)
    logger.error("Compute uncertainty metrics")
    logger.error("Compute mu pred. entropy")
    error, mu_entropy, pred_y = sess.run(predict_cross_entropy(y_test, test_preds))
    logger.error("Compute variation ratios")
    voted_pred = K.get_session().run(voting(predictions))
    logger.error("Compute beta pred. entropy")
    sampling_entropy_gal = sess.run(predict_dirichlet_entropy_gal(predictions))
    logger.error("Compute rejection measures")
    rejection_measures = np.array(
        [list(get_rejection_measures(np.argmax(pred_y, axis=1), np.argmax(y_test, axis=1),
                                     np.argsort(sampling_entropy_gal),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_baseline = np.array(
        [list(get_rejection_measures(np.argmax(pred_y, axis=1), np.argmax(y_test, axis=1), np.argsort(mu_entropy),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    rejection_measures_voting = np.array(
        [list(get_rejection_measures(np.argmax(pred_y, axis=1), np.argmax(y_test, axis=1), np.argsort(voted_pred),
                                     rejection_point))
         for rejection_point in range(1, pred_y.shape[0] - 10)])
    logger.error("Export results")
    with open(args.output_file, 'wb') as file:
        pickle.dump((mu_entropy, error, voted_pred, sampling_entropy_gal, rejection_measures,
                     rejection_measures_baseline, rejection_measures_voting,
                     y_test), file)
    logger.info("Done")
