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


def load_data(training_file, val_file):
    train_df = pd.read_csv(training_file, sep='\t', header=0)
    train_df = train_df.dropna()
    val_df = pd.read_csv(val_file, sep='\t', header=0)
    val_df = val_df.dropna()
    return train_df, val_df


def preprocess(df):
    df['clean_text'] = df['sentence'].apply(process_text)
    df = df[df['clean_text'].map(lambda d: len(d)) > 0]
    return df


def load_w2v_model(google_w2v_file):
    google_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(google_w2v_file, binary=True)
    return dict(zip(google_model.wv.index2word, google_model.wv.syn0))


def build_model():
    input = Input(shape=(OUTPUT_LAYER_SIZE,))
    output = Dense(NUM_CLASSES, activation='softmax', name='output_layer')(input)
    return Model(inputs=[input], outputs=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load the job offers from different sources to a common ES index")
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs to train uncertainty model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--google_w2v_file', type=str, default='GoogleNews-vectors-negative300.bin.gz',
                        help='google_w2v_file')
    parser.add_argument('--output_model_file', type=str, default='sst2_model',
                        help='file to dump the generated model')
    parser.add_argument('--output_history_file', type=str, default='sst2_history',
                        help='file to dump the generated data')
    parser.add_argument('--training_file', type=str, default='training.tsv',
                        help='file with the training set')
    parser.add_argument('--val_file', type=str, default='val.tsv',
                        help='file with the training set')

    args = parser.parse_args()
    logger = get_logger()
    logger.info("Load data")
    train_df, val_df = load_data(args.training_file, args.val_file)
    logger.info("Preprocess train data")
    train_df = preprocess(train_df)
    logger.info("Preprocess val data")
    val_df = preprocess(val_df)
    logger.info("Load w2v vector")
    w2v = load_w2v_model(args.google_w2v_file)

    NUM_CLASSES = 2

    logger.info("Prepare features")
    # Creating the features (tf-idf weights) for the processed text
    w2v_vectorizer = MeanEmbeddingVectorizer(w2v)
    X_train = w2v_vectorizer.transform(train_df['clean_text'])  # features
    y_train = train_df['label'].values  # target
    y_train = np.eye(NUM_CLASSES)[y_train.astype(dtype=np.int32)]
    X_val = w2v_vectorizer.transform(val_df['clean_text'])  # features
    y_val = val_df['label'].values  # target
    y_val = np.eye(NUM_CLASSES)[y_val.astype(dtype=np.int32)]

    OUTPUT_LAYER_SIZE = X_train.shape[1]
    logger.info("Build the model")
    model = build_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    logger.info("Train the model")
    training_history = model.fit([X_train],
                                 y_train,
                                 batch_size=128,
                                 epochs=80,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=([X_val],
                                                  y_val))
    logger.info("Save the history")
    with open(args.output_history_file, 'wb') as file:
        pickle.dump(training_history.history, file)
    logger.info("Save the model")
    tf.keras.models.save_model(
        model,
        args.output_model_file,
        overwrite=True,
        include_optimizer=True,
        save_format=None
    )
    logger.info("Done")
