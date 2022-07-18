import argparse
import json
import time

import numpy as np

from SentenceGetter import SentenceGetter
from dataprocess import buildDataset
from io_utils import getPathToFile
from training_preparation import constructModel, getWordEmbeddings, getTagMap, getOneHotEncoding

if __name__ == "__main__":

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_length', type=int, default=80, help='the sentence length')
    parser.add_argument('--number_epochs', type=int, default=50, help='the number of epochs while training')
    parser.add_argument('--filter_dataset', type=bool, default=False, help='boolean to in- or exclude dataset filtering')
    parser.add_argument('--path', default="../data/train", help='the path to the training dataset')
    args = parser.parse_args()

    PATH = args.path
    SENTENCE_LENGTH = args.sentence_length
    NUMBER_EPOCHS = args.number_epochs
    FILTER_DATASET = args.filter_dataset
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data, FILTER_DATASET)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    tag2idx = getTagMap()
    n_tags = len(tag2idx)
    y_train = getOneHotEncoding(tag2idx, sentences, n_tags, SENTENCE_LENGTH)

    X_train = getWordEmbeddings(sentences, SENTENCE_LENGTH)

    model = constructModel(n_tags, SENTENCE_LENGTH)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(np.array(X_train), np.array(y_train), batch_size=32, epochs=NUMBER_EPOCHS, validation_split=0.1, verbose=1)

    path_to_file = getPathToFile(SENTENCE_LENGTH, NUMBER_EPOCHS, FILTER_DATASET)
    model.save(path_to_file)

    model.summary()

    end = time.time()
    print("Training of model took " + str(end - start) + "s")
