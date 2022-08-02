import numpy as np
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import tensorflow as tf

from nltk.tokenize import word_tokenize, sent_tokenize
from processing.training_preparation import getTagMap, getWordEmbeddingsForPrediction


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return ""


def getWordLemmas(documents):
    all_lemmas = []
    lemmatizer = WordNetLemmatizer()

    tokenized_sent = [item for doc in documents for item in sent_tokenize(doc["text"])]
    tokenized_docs = [word_tokenize(sent) for sent in tokenized_sent]

    for sent in tokenized_docs:
        pos_tags = [get_wordnet_pos(tup[1]) for tup in pos_tag(sent)]
        lemmas = [lemmatizer.lemmatize(t, pos=pos_tags[ind]).lower() if pos_tags[ind] != "" else t.lower()
                  for ind, t in enumerate(sent)]
        all_lemmas.append(lemmas)
    return all_lemmas


def predictCategories(model, X_test, all_lemmas, tag2idx):
    all_tags = []

    for i in range(len(X_test)):
        sentence = all_lemmas[i]
        words = [w for w in sentence]
        y_pred = model.predict(np.array([X_test[i]]))
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = y_pred[0][:len(words)]

        tags = [list(tag2idx.keys())[pred] for pred in y_pred]
        all_tags.append(tags)

    return all_tags


def flattenNestedList(nestedList):
    return [item for sublist in nestedList for item in sublist]


def createCodes(all_lemmas, all_tags):
    code_index = 0
    codes = []

    for index, lemma in enumerate(all_lemmas):
        if (all_tags[index] != "None") and (all_tags[index] != "_"):
            # create code and append to list of codes
            code = {
                "tokens": [
                    index
                ],
                "name": lemma,
                "tore": all_tags[index],
                "index": code_index,
                "relationship_memberships": []
            }

            codes.append(code)
            code_index += 1
    return codes


def classifyDataset(documents, SENTENCE_LENGTH=80, MODEL_PATH="../model/80/model_2layers_50e_unfiltered.h5"):

    all_lemmas = getWordLemmas(documents)

    tag2idx = getTagMap()
    X_test = getWordEmbeddingsForPrediction(all_lemmas, SENTENCE_LENGTH)

    model = tf.keras.models.load_model(MODEL_PATH)
    all_tags = predictCategories(model, X_test, all_lemmas, tag2idx)

    all_tags = flattenNestedList(all_tags)
    all_lemmas = flattenNestedList(all_lemmas)

    return createCodes(all_lemmas, all_tags)
