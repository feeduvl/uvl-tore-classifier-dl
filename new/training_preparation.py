import json

from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.python.keras import Model, Input

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import gensim.downloader as api

from new.io_utils import saveContentToFile
import numpy as np


def constructModel(n_tags, sentence_length):
    input = Input(shape=(sentence_length, 100))
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, input_shape=(sentence_length,100)))(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(model)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    return Model(input, out)

def constructAndSaveTagMap(tags):
    tag2idx = {t: i for i, t in enumerate(tags)}

    saveContentToFile("../data/generated", "tag2idx.json", tag2idx)
    saveContentToFile("../data/generated", "tags.json", tags)
    return tag2idx

def getTags(dataset):
    tags = list(set(dataset["tag"].values))
    tags.append("_")
    return tags


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [np.zeros(100).tolist()]*(target_len - len(some_list))


def getWordEmbeddings(sentences, max_len, save=False):
    model_glove_twitter = api.load("glove-twitter-100")
    X = [[(model_glove_twitter[w[0]].tolist() if (w[0] in model_glove_twitter) else np.zeros(100).tolist()) for w in s] for s in sentences]
    X = [pad_or_truncate(s, max_len) for s in X]
    if save:
        # TODO: Change back
        saveContentToFile("../data/generated", "embeddings.json", X)
        # saveContentToFile("../data/generated/test", "embeddings.json", X)
    return X


def loadWordEmbedding():
    # TODO: Change back
    f = open("../data/generated" + "/embeddings.json")
    # f = open("../data/generated/test" + "/embeddings.json")
    return json.load(f)


def getXAndy(tag2idx, sentences, n_tags, sentence_length):
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=sentence_length, sequences=y, padding="post", value=tag2idx["_"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    return y