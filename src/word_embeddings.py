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


def mapTagToCorrectSize(y_training, len_sentence):
    y_train_filled = []
    for tags in y_training:
        correct_length = np.zeros(len_sentence, dtype=np.int8)
        correct_length.fill(0)
        for i in range(len(tags)):
            correct_length[i] = tags[i]
        y_train_filled.append(correct_length.tolist())
    return y_train_filled


def getWordEmbeddings(anno_data, tore_dict):
    x_train, y_train = getTrainingSets(anno_data, tore_dict)

    embedding_dimension = 100
    length_of_sentence = 80

    y_train = mapTagToCorrectSize(y_train, length_of_sentence)
    path = "../data"
    filename = "y_train.json"

    saveContentToFile(path, filename, y_train)

    sentence_matrix = []
    for sentence in x_train:
        gensim_weight_matrix = np.zeros((length_of_sentence, embedding_dimension))
        for i in range(len(sentence)):
            if sentence[i] in model_glove_twitter:
                gensim_weight_matrix[i] = model_glove_twitter[sentence[i]]
            else:
                gensim_weight_matrix[i] = np.zeros(embedding_dimension)
        gensim_list = gensim_weight_matrix.tolist()
        sentence_matrix.append(gensim_list)
    return sentence_matrix

if __name__ == "__main__":

    # 100 refers to dimensionality of the vector, 100 comes closest to 128 dimensions in paper
    model_glove_twitter = api.load("glove-twitter-100")

    f = open("../data/sentences.json")
    my_anno_data = json.load(f)
    tore_dictionary = getToreDict()

    embeddings_for_sentences = getWordEmbeddings(my_anno_data, tore_dictionary)

    PATH = "../data"
    FILENAME = "embeddings.json"

    saveContentToFile(PATH, FILENAME, embeddings_for_sentences)

