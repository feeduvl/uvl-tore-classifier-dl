import json
from os.path import exists

import gensim.downloader as api
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.python.keras import Model, Input

from processing.io_utils import saveContentToFile


def constructModel(n_tags, sentence_length):
    input = Input(shape=(sentence_length, 100))
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, input_shape=(sentence_length,100)))(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    return Model(input, out)


def getTagMap(isLocal=False):
    if isLocal:
        f = open("../data/global" + "/tag2idx.json")
    else:
        f = open("data/global" + "/tag2idx.json")
    return json.load(f)


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [np.zeros(100).tolist()]*(target_len - len(some_list))


def getPathWordEmbedding(sen_len, isTest, dataset_filtered):
    embedding_path = "../data/generated"
    if isTest:
        embedding_path = embedding_path + "/test"

    if dataset_filtered:
        embedding_filename = "/embeddings_filtered" + str(sen_len) + ".json"
    else:
        embedding_filename = "/embeddings_unfiltered" + str(sen_len) + ".json"
    return embedding_path, embedding_filename


def getWordEmbeddings(sentences, sen_len, isTest=False, dataset_filtered=False):

    embedding_path, embedding_filename = getPathWordEmbedding(sen_len, isTest, dataset_filtered)

    if exists(embedding_path + embedding_filename):
        print("Word embeddings already exist, load from file")
        f = open(embedding_path + embedding_filename)
        return json.load(f)
    else:
        print("Calculating word embeddings, this might take a while")
        model_glove_twitter = api.load("glove-twitter-100")

        X = [[(model_glove_twitter[w[0]].tolist() if (w[0] in model_glove_twitter) else np.zeros(100).tolist()) for w in s] for s in sentences]
        X = [pad_or_truncate(s, sen_len) for s in X]

        saveContentToFile(embedding_path, embedding_filename, X)
        return X


def getWordEmbeddingsForPrediction(sentences, sen_len):

    print("Calculating word embeddings, this might take a while")
    model_glove_twitter = api.load("glove-twitter-100")

    X = [[(model_glove_twitter[w].tolist() if (w in model_glove_twitter) else np.zeros(100).tolist()) for w in s] for s in sentences]
    X = [pad_or_truncate(s, sen_len) for s in X]

    return X


def getOneHotEncoding(tag2idx, sentences, n_tags, sentence_length):
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=sentence_length, sequences=y, padding="post", value=tag2idx["_"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    return y
