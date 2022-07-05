import json

import numpy as np
from matplotlib import pyplot as plt

from new.SentenceGetter import SentenceGetter
from new.dataprocess import buildDataset
from new.training_preparation import constructModel, loadWordEmbedding, constructAndSaveTagMap, getTags, getXAndy, getWordEmbeddings

if __name__ == "__main__":

    PATH = "../data/train"
    SENTENCE_LENGTH = 60
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    tags = getTags(dataset)
    n_tags = len(tags)

    tag2idx = constructAndSaveTagMap(tags)
    y_train = getXAndy(tag2idx, sentences, n_tags, SENTENCE_LENGTH)

    # Uncomment first and comment second, when building word embeddings
    # getWordEmbeddings(sentences, SENTENCE_LENGTH, True)
    X_train = loadWordEmbedding()

    model = constructModel(n_tags, SENTENCE_LENGTH)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=25, validation_split=0.1, verbose=1)

    model.save('../emb_model/' + str(SENTENCE_LENGTH) + '/model02_25e.h5')
    model.summary()

    plt.plot(history.history['accuracy'], c='b', label='train accuracy')
    plt.plot(history.history['val_accuracy'], c='r', label='validation accuracy')
    plt.legend(loc='lower right')
    plt.show()
