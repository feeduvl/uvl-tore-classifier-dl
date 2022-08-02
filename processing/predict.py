# import argparse
# import itertools
# import json
#
# import numpy as np
# from nltk import pos_tag
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from sklearn.metrics import classification_report, accuracy_score
#
# import tensorflow as tf
#
# from SentenceGetter import SentenceGetter
# from calculate_stats import calculatePresRecall
# from dataprocess import buildDataset
# from training_preparation import getWordEmbeddings, getTagMap, getOneHotEncoding, getWordEmbeddingsForPrediction
#
# from nltk.tokenize import word_tokenize, sent_tokenize
#
#
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     if treebank_tag.startswith('V'):
#         return wordnet.VERB
#     if treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     if treebank_tag.startswith('R'):
#         return wordnet.ADV
#     return ""
#
#
# if __name__ == "__main__":
#
#     # Remove this!!
#     f = open("../tmp/ds.json")
#     data = json.load(f)
#
#     documents = data["documents"]
#     dataset_name = data["name"]
#     annotation_name = "Test anno"
#     create = True
#
#     # Remove before this!
#
#     # Sentence length of the current model
#     SENTENCE_LENGTH = 80
#     MODEL_PATH = "../model/80/model_2layers_50e_unfiltered.h5"
#
#     all_tags = []
#     sentences = []
#
#     lemmatizer = WordNetLemmatizer()
#
#     tokenized_sent = [item for doc in documents for item in sent_tokenize(doc["text"])]
#     tokenized_docs = [word_tokenize(sent) for sent in tokenized_sent]
#
#     all_words = [item for sublist in tokenized_docs for item in sublist]
#
#     all_lemmas = []
#     for sent in tokenized_docs:
#         pos_tags = [get_wordnet_pos(tup[1]) for tup in pos_tag(sent)]
#         lemmas = [lemmatizer.lemmatize(t, pos=pos_tags[ind]).lower() if pos_tags[ind] != "" else t.lower()
#                   for ind, t in enumerate(sent)]
#         all_lemmas.append(lemmas)
#
#
#     tag2idx = getTagMap()
#     n_tags = len(tag2idx)
#     # X_test = getWordEmbeddingsForPrediction(all_lemmas, SENTENCE_LENGTH)
#     X_test = getWordEmbeddingsForPrediction()
#
#     model = tf.keras.models.load_model(MODEL_PATH)
#
#     for i in range(len(X_test)):
#         sentence = tokenized_docs[i]
#         words = [w for w in sentence]
#         y_pred = model.predict(np.array([X_test[i]]))
#         y_pred = np.argmax(y_pred, axis=-1)
#         y_pred = y_pred[0][:len(words)]
#
#         tags = [list(tag2idx.keys())[pred] for pred in y_pred]
#         all_tags.append(tags)
#
#     all_tags = [item for sublist in all_tags for item in sublist]
#     all_lemmas = [item for sublist in all_lemmas for item in sublist]
#
#     code_index = 0
#     codes = []
#
#     for index, lemma in enumerate(all_lemmas):
#         if (all_tags[index] != "None") and (all_tags[index] != "_"):
#             # create code and append to list of codes
#             code = {
#                 "tokens": [
#                     index
#                 ],
#                 "name": lemma,
#                 "tore": all_tags[index],
#                 "index": code_index,
#                 "relationship_memberships": []
#             }
#
#             codes.append(code)
#             code_index += 1
#
#     for bli in codes:
#         print(bli)
