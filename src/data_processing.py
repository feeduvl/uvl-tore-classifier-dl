import json
import os
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


def getLemmaOfTokensWAndInitilizeCodes(list_of_tokens):
    tokens_with_code = []
    for i in range(len(list_of_tokens)):
        token_with_code = {
            "lemma": list_of_tokens[i]["lemma"],
            "tag": None
        }
        tokens_with_code.append(token_with_code)
    return tokens_with_code


def fillToreCategories(list_of_tokens_with_code, list_of_codes):
    for i in range(len(list_of_codes)):
        # List of tokens could be empty
        if len(list_of_codes[i]["tokens"]) > 0:
            for j in range(len(list_of_codes[i]["tokens"])):
                token = list_of_codes[i]["tokens"][j]
                list_of_tokens_with_code[token]["tag"] = list_of_codes[i]["tore"]
    return list_of_tokens_with_code


def constructFeatureVector():
    all_tokens = my_anno_data["tokens"]
    all_codes = my_anno_data["codes"]

    all_tokens_with_code = getLemmaOfTokensWAndInitilizeCodes(all_tokens)
    all_tokens_with_code = fillToreCategories(all_tokens_with_code, all_codes)
    return all_tokens_with_code


def constructDocuments(feature_vecs):
    documents = []
    for i in range(len(all_docs)):
        start = all_docs[i]["begin_index"]
        end = all_docs[i]["end_index"]
        documents.append(feature_vecs[start:end])
    return documents


def getSentencesForDocument(document):
    sentences = []
    sentence = []
    length_of_doc = len(document)
    for i in range(length_of_doc):
        if document[i]["lemma"] == "#":
            if (i + 2) < length_of_doc:
                if (document[i + 1]["lemma"] == "#") and (document[i + 2]["lemma"] == "#"):
                    if len(sentence) != 0:
                        sentences.append(sentence)
                    sentence = []
        elif (document[i]["lemma"] == ".") or (document[i]["lemma"] == "!") or (document[i]["lemma"] == "?"):
            if len(sentence) != 0:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(document[i])
    return sentences


def normalizeSentences(sentences_not_normalized):
    all_sent_normalized = []
    for sent in sentences_not_normalized:
        sent_without_sw = [token for token in sent if not token["lemma"] in stopwords.words('english')]
        sent_normalized = [token for token in sent_without_sw if not re.sub(r"[^a-zA-Z0-9 ]", "", token["lemma"]) == ""]
        all_sent_normalized.append(sent_normalized)
    return all_sent_normalized

def saveSentencesToFile(path, file, normlized_sentences):
    if os.path.isfile(path) and os.access(path, os.R_OK):
        # checks if file exists
        print("File exists and is readable")
    else:
        print("Either file is missing or is not readable, creating file...")
        with open(os.path.join(path, file), 'w') as fout:
            json.dump(normlized_sentences, fout)


if __name__ == "__main__":

    f = open("../data/anno.json")
    my_anno_data = json.load(f)

    all_docs = my_anno_data["docs"]

    # Construct feature vectors from tokens and codes, and sort them by document
    feature_vectors = constructFeatureVector()
    docs = constructDocuments(feature_vectors)
    counter = 0

    for doc in docs:
        sentences = getSentencesForDocument(doc)
        sentences_normalized = normalizeSentences(sentences)

        PATH = "../data/documents"
        FILENAME = "doc" + str(counter) + ".json"

        saveSentencesToFile(PATH, FILENAME, sentences_normalized)
        counter += 1


