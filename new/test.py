import json

import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import tensorflow as tf

from new.SentenceGetter import SentenceGetter
from new.dataprocess import buildDataset

if __name__ == "__main__":
    PATH = "../data/test"
    SENTENCE_LENGTH = 40
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    maxlen = max([len(s) for s in sentences])
    print('Maximum sequence length:', maxlen)

    words = list(set(dataset["word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    print("Number of words: ", n_words)
    tags = list(set(dataset["tag"].values))
    n_tags = len(tags)
    print("Number of tags: ", n_tags)

    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    X_test = [[word2idx[w[0]] for w in s] for s in sentences]
    X_test = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=X_test, padding="post", value=n_words - 1)
    y_test = [[tag2idx[w[1]] for w in s] for s in sentences]
    y_test = pad_sequences(maxlen=SENTENCE_LENGTH, sequences=y_test, padding="post", value=tag2idx["None"])
    y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

    model = tf.keras.models.load_model('../model/my_model5.h5')

    i = 0
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
    for w, pred in zip(X_test[i], p[0]):
        print("{:14}: {}".format(words[w], tags[pred]))