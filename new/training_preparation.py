from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.python.keras import Model, Input

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from new.io_utils import saveContentToFile


def constructModel(n_words, n_tags, sentence_length):
    input = Input(shape=(sentence_length,))
    model = Embedding(input_dim=n_words, output_dim=sentence_length, input_length=sentence_length)(input)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1))(model)
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    return Model(input, out)

def constructAndSaveWordAndTagMaps(words, tags):
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    saveContentToFile("../data/generated", "word2idx.json", word2idx)
    saveContentToFile("../data/generated", "tag2idx.json", tag2idx)
    return word2idx, tag2idx

def getWordsAndTags(dataset):
    words = list(set(dataset["word"].values))
    words.append("ENDPAD")
    tags = list(set(dataset["tag"].values))
    tags.append("_")
    return words, tags


def getXAndy(word2idx, tag2idx, sentences, n_words, n_tags, sentence_length):
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=sentence_length, sequences=X, padding="post", value=n_words - 1)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=sentence_length, sequences=y, padding="post", value=tag2idx["_"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    return X, y