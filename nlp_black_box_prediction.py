import argparse
import pickle

import gensim

import pandas as pd
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf

from tools import process_text, MeanEmbeddingVectorizer
from tools import get_logger


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
    return df


def load_w2v_model(google_w2v_file):
    google_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(google_w2v_file, binary=True)
    return dict(zip(google_model.wv.index2word, google_model.wv.syn0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--google_w2v_file', type=str, default='GoogleNews-vectors-negative300.bin.gz',
                        help='google_w2v_file')
    parser.add_argument('--bb_model_file', type=str, default='sst2_model',
                        help='file to dump the generated model')
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
    logger.info("Load w2v vector")
    w2v = load_w2v_model(args.google_w2v_file)

    NUM_CLASSES = 2

    logger.info("Prepare features")
    # Creating the features (tf-idf weights) for the processed text
    w2v_vectorizer = MeanEmbeddingVectorizer(w2v)
    X_train = w2v_vectorizer.transform(train_df['clean_text'])  # features
    X_val = w2v_vectorizer.transform(val_df['clean_text'])  # features
    X_test = w2v_vectorizer.transform(test_df['clean_text'])  # features
    logger.info("Load the model")
    model = tf.keras.models.load_model(args.bb_model_file)
    logger.info("Make predictions")
    train_preds = model.predict(X_train, batch_size=1024)
    val_preds = model.predict(X_val, batch_size=1024)
    test_preds = model.predict(X_test, batch_size=1024)
    logger.info("Save the predictions")
    with open(args.training_preds, 'wb') as file:
        pickle.dump(train_preds, file)
    with open(args.val_preds, 'wb') as file:
        pickle.dump(val_preds, file)
    with open(args.test_preds, 'wb') as file:
        pickle.dump(test_preds, file)
    logger.info("Done")
