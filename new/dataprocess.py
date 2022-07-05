import re

from nltk.corpus import stopwords
import pandas as pd


def getLemmaOfTokensWAndInitilizeCodes(list_of_tokens):
    tokens_with_code = []
    for i in range(len(list_of_tokens)):
        token_with_code = {
            "word": list_of_tokens[i]["lemma"],
            "index": list_of_tokens[i]["index"],
            "sentence_idx": None,
            "tag": "None"
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

def filter_tokens(tokens_with_code):
    tokens_with_code = [token for token in tokens_with_code if not token["word"] in stopwords.words('english')]
    tokens_with_code = [token for token in tokens_with_code if not re.sub(r"[^a-zA-Z0-9 ]", "", token["word"]) == ""]
    return tokens_with_code

def constructFeatureVector(data_vector):
    all_tokens = data_vector["tokens"]
    all_codes = data_vector["codes"]

    all_tokens_with_code = getLemmaOfTokensWAndInitilizeCodes(all_tokens)
    all_tokens_with_code = fillToreCategories(all_tokens_with_code, all_codes)

    return all_tokens_with_code


def constructDocuments(feature_vecs, all_docs):
    documents = []
    for i in range(len(all_docs)):
        start = all_docs[i]["begin_index"]
        end = all_docs[i]["end_index"]
        documents.append(feature_vecs[start:end])
    return documents


def getSentencesForDocument(documents):
    sentencesounter = 0
    sentences = []
    for document in documents:
        length_of_doc = len(document)
        is_used = False
        for i in range(length_of_doc):
            if document[i]["word"] == "#":
                if (i + 2) < length_of_doc:
                    if (document[i + 1]["word"] == "#") and (document[i + 2]["word"] == "#"):
                        if is_used:
                            sentencesounter += 1
                            is_used = False
            elif (document[i]["word"] == ".") or (document[i]["word"] == "!") or (document[i]["word"] == "?"):
                if is_used:
                    sentencesounter += 1
                    is_used = False
            else:
                document[i]["sentence_idx"] = sentencesounter
                is_used = True
            sentences.append(document[i])
    return sentences

def buildDataset(data):
    all_docs = data["docs"]

    feature_vectors = constructFeatureVector(data)
    # In between documents there might not be punctuation, so this is done manually
    docs = constructDocuments(feature_vectors, all_docs)
    # Sentences are not only separated by normal sentence separators, but also by '###'
    ds = getSentencesForDocument(docs)

    return pd.DataFrame(ds)