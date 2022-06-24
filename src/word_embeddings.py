import json

import gensim.downloader as api
import numpy as np


def getToreDict():
    f = open("../data/tore.json")
    tore = json.load(f)
    tores = tore["tores"]
    tores.append(None)
    tore_dict = {}

    for i in range(len(tores)):
        tore_dict[tores[i]] = i
    return tore_dict

if __name__ == "__main__":

    # 100 refers to dimensionality of the vector, 100 comes closest to 128 dimensions in paper
    model_glove_twitter = api.load("glove-twitter-100")

    EMB_DIM = 100
    MAX_LEN = 80

    f = open("../data/sentences.json")
    my_anno_data = json.load(f)

    tore_dictionary = getToreDict()

    x_train = []
    y_train = []
    for sentence in my_anno_data:
        only_words = []
        only_tags = []
        for tokens in sentence:
            only_words.append(tokens["lemma"])
            encoded_tag = tore_dictionary[tokens["tag"]]
            only_tags.append(encoded_tag)
        x_train.append(only_words)
        y_train.append(only_tags)

    sentence_matrix = []
    for sentence in x_train:
        gensim_weight_matrix = np.zeros((MAX_LEN, EMB_DIM))
        for i in range(len(sentence)):
            if sentence[i] in model_glove_twitter:
                gensim_weight_matrix[i] = model_glove_twitter[sentence[i]]
            else:
                gensim_weight_matrix[i] = np.zeros(100)
        sentence_matrix.append(gensim_weight_matrix)
        print(len(gensim_weight_matrix))

    print(len(sentence_matrix))
