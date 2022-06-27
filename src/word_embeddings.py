import json

import gensim.downloader as api
import numpy as np

from src.io_utils import saveContentToFile


def getToreDict():
    f = open("../data/tore.json")
    tore = json.load(f)
    tores = tore["tores"]
    tores.append(None)
    tore_dict = {}

    for i in range(len(tores)):
        tore_dict[tores[i]] = i + 1
    return tore_dict


def getTrainingSets(anno_data, tore_dict):
    x_training = []
    y_training = []
    for sent in anno_data:
        only_words = []
        only_tags = []
        for tokens in sent:
            only_words.append(tokens["lemma"])
            encoded_tag = tore_dict[tokens["tag"]]
            only_tags.append(encoded_tag)
        x_training.append(only_words)
        y_training.append(only_tags)
    return x_training, y_training


def mapFeatureToCorrectSize(x_train, len_sentence):
    x_train_filled = []
    for sentence in x_train:
        filler = []
        for i in range(len_sentence):
            if len(sentence) > i:
                filler.append(sentence[i])
            else:
                filler.append("")
        x_train_filled.append(filler)
    return x_train_filled


def mapTagToCorrectSize(y_train, len_sentence):
    y_train_filled = []
    for tags in y_train:
        correct_length = np.zeros(len_sentence, dtype=np.int8)
        correct_length.fill(0)
        for i in range(len(tags)):
            correct_length[i] = tags[i]
        y_train_filled.append(correct_length.tolist())
    return y_train_filled


def saveWordEmbeddings(x_train, emb_dim, len_sentence):
    # 100 refers to dimensionality of the vector, 100 comes closest to 128 dimensions in paper
    model_glove_twitter = api.load("glove-twitter-100")

    sentence_matrix = []
    for sentence in x_train:
        gensim_weight_matrix = np.zeros((len_sentence, emb_dim))
        for i in range(len(sentence)):
            if sentence[i] in model_glove_twitter:
                gensim_weight_matrix[i] = model_glove_twitter[sentence[i]]
            else:
                gensim_weight_matrix[i] = np.zeros(emb_dim)
        gensim_list = gensim_weight_matrix.tolist()
        sentence_matrix.append(gensim_list)

    PATH = "../data"
    FILENAME = "embeddings.json"

    saveContentToFile(PATH, FILENAME, sentence_matrix)


def getFeatureVectorsAndSave(anno_data, tore_dict, len_sentence):
    x_train, y_train = getTrainingSets(anno_data, tore_dict)

    x_train = mapFeatureToCorrectSize(x_train, len_sentence)
    y_train = mapTagToCorrectSize(y_train, len_sentence)
    path = "../data"
    filename1 = "features.json"
    filename2 = "tags.json"

    saveContentToFile(path, filename1, x_train)
    saveContentToFile(path, filename2, y_train)
    return x_train, y_train


if __name__ == "__main__":

    f = open("../data/sentences.json")
    my_anno_data = json.load(f)
    tore_dictionary = getToreDict()

    EMB_DIM = 100
    LEN_SENTENCE = 80

    x_training, y_training = getFeatureVectorsAndSave(my_anno_data, tore_dictionary, LEN_SENTENCE)
    saveWordEmbeddings(x_training, EMB_DIM, LEN_SENTENCE)


