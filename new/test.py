import json

import numpy as np
from sklearn import metrics

import tensorflow as tf

from new.SentenceGetter import SentenceGetter
from new.dataprocess import buildDataset
from new.training_preparation import getXAndy, getWordEmbeddings, loadWordEmbedding

# TODO: Rework!!
if __name__ == "__main__":
    PATH = "../data/test"
    SENTENCE_LENGTH = 60
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    f = open("../data/generated/tags.json")
    tags = json.load(f)
    n_tags = len(tags)

    f = open("../data/generated/tag2idx.json")
    tag2idx = json.load(f)
    y_test = getXAndy(tag2idx, sentences, n_tags, SENTENCE_LENGTH)
    # X_test = getWordEmbeddings(sentences, SENTENCE_LENGTH, True)
    X_test = loadWordEmbedding()

    model = tf.keras.models.load_model('../emb_model/60/model02_25e.h5')

    for i in range(len(X_test)):
        sentence = sentences[i]
        words = [w[0] for w in sentence]
        print("round" + str(i))
        p = model.predict(np.array([X_test[i]]))
        p = np.argmax(p, axis=-1)
        y_true = np.argmax(y_test[i], axis=-1)
        # print(p)
        # print(y_true)
        # print("{:14}: {}".format("True", "Pred"))
        # for t, pred in zip(y_true, p[0]):
        #     print("{:14}: {}".format(tags[t], tags[pred]))
        print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
        for w, t, pred in zip(words, y_true, p[0]):
            print("{:14}: {}, {}".format(w, tags[t], tags[pred]))

