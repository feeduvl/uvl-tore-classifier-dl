import json

import numpy as np
from sklearn import metrics

import tensorflow as tf

from new.SentenceGetter import SentenceGetter
from new.dataprocess import buildDataset
from new.training_preparation import getWordsAndTags, getXAndy

if __name__ == "__main__":
    PATH = "../data/test"
    SENTENCE_LENGTH = 40
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    # maxlen = max([len(s) for s in sentences])
    # print('Maximum sequence length:', maxlen)

    f = open("../data/generated/words.json")
    words = json.load(f)
    f = open("../data/generated/tags.json")
    tags = json.load(f)

    n_words = len(words)
    n_tags = len(tags)

    f = open("../data/generated/word2idx.json")
    word2idx = json.load(f)
    f = open("../data/generated/tag2idx.json")
    tag2idx = json.load(f)
    X_test, y_test = getXAndy(word2idx, tag2idx, sentences, n_words, n_tags, SENTENCE_LENGTH)


    model = tf.keras.models.load_model('../new_model/40/model05_3e.h5')

    i = 0
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    y_true = np.argmax(y_test[i], axis=-1)
    print(p)
    print(y_true)
    print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
    for w, t, pred in zip(X_test[i], y_true, p[0]):
        print("{:14}: {}, {}".format(words[w], tags[t], tags[pred]))

