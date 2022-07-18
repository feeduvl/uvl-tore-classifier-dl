import argparse
import json

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf

from SentenceGetter import SentenceGetter
from calculate_stats import calculatePresRecall
from dataprocess import buildDataset
from training_preparation import getWordEmbeddings, getTagMap, getOneHotEncoding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_length', type=int, default=80, help='the sentence length')
    parser.add_argument('--path', default="../data/test", help='the path to the training dataset')
    parser.add_argument('--path_to_model', default="../model/80/model_2layers_50e_unfiltered.h5", help='the path to the currently used model')
    args = parser.parse_args()

    PATH = args.path
    SENTENCE_LENGTH = args.sentence_length
    f = open(PATH + "/anno.json")
    data = json.load(f)

    dataset = buildDataset(data)
    getter = SentenceGetter(dataset)
    sentences = getter.sentences
    print("Number of sentences: ", len(sentences))

    tag2idx = getTagMap()
    n_tags = len(tag2idx)
    y_test = getOneHotEncoding(tag2idx, sentences, n_tags, SENTENCE_LENGTH)
    X_test = getWordEmbeddings(sentences, SENTENCE_LENGTH, True)

    model = tf.keras.models.load_model(args.path_to_model)

    y_true_global = []
    y_pred_global = []

    for i in range(len(X_test)):
        sentence = sentences[i]
        words = [w[0] for w in sentence]
        y_pred = model.predict(np.array([X_test[i]]))
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_test[i], axis=-1)
        y_pred = y_pred[0][:len(words)]
        y_true = y_true[:len(words)]

        for j in y_true:
            y_true_global.append(j)

        for j in y_pred:
            y_pred_global.append(j)

    # From here on, only metrics
    print(classification_report(y_true_global, y_pred_global, digits=4))
    all_acc = accuracy_score(y_true_global, y_pred_global)
    print("Accuracy including None: " + str(all_acc))

    # Calculate accuracy excluding None
    count = 0
    leng = 0
    for i in range(len(y_true_global)):
        if y_true_global[i] != 0:
            leng += 1
            if y_true_global[i] == y_pred_global[i]:
                count += 1

    print("Accuracy excluding None: " + str(count/leng))


    recall, precision = calculatePresRecall(y_true_global, y_pred_global)

    f1_score = []
    for i in range(len(recall)):
        if (precision[i] + recall[i]) != 0:
            f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        else:
            f1_score.append(0.0)

    print("\nMetrics excluding categories not included in test dataset, with None: ")
    print("\tPrecision: " + str(np.mean(precision)))
    print("\tRecall: " + str(np.mean(recall)))
    print("\tF1-Score: " + str(np.mean(f1_score)))

    recall.pop(0)
    precision.pop(0)
    f1_score.pop(0)

    print("\nMetrics excluding categories not included in test dataset, without None: ")
    print("\tPrecision: " + str(np.mean(precision)))
    print("\tRecall: " + str(np.mean(recall)))
    print("\tF1-Score: " + str(np.mean(f1_score)))

    recall, precision = calculatePresRecall(y_true_global, y_pred_global)

    recall.append(1.0)
    recall.append(1.0)
    recall.append(1.0)
    recall.append(1.0)
    precision.append(1.0)
    precision.append(1.0)
    precision.append(1.0)
    precision.append(1.0)

    f1_score = []
    for i in range(len(recall)):
        if (precision[i] + recall[i]) != 0:
            f1_score.append((2 * precision[i] * recall[i]) / (precision[i] + recall[i]))
        else:
            f1_score.append(0.0)

    print("\nMetrics all categories, with None: ")
    print("\tPrecision: " + str(np.mean(precision)))
    print("\tRecall: " + str(np.mean(recall)))
    print("\tF1-Score: " + str(np.mean(f1_score)))

    recall.pop(0)
    precision.pop(0)
    f1_score.pop(0)

    print("\nMetrics all categories, without None: ")
    print("\tPrecision: " + str(np.mean(precision)))
    print("\tRecall: " + str(np.mean(recall)))
    print("\tF1-Score: " + str(np.mean(f1_score)))