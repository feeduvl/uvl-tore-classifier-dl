import json

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers import TimeDistributed
import tensorflow as tf
import numpy as np
from matplotlib import pyplot


def loadEmbeddingsAndTags(path):
    f = open(path + "/embeddings.json")
    embeddings = json.load(f)
    f = open(path + "/tags.json")
    tags = json.load(f)
    return embeddings, tags


def createModel(emb_dim, sen_len, class_num):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(sen_len, emb_dim))))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(TimeDistributed(Dense(class_num, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    return model


if __name__ == "__main__":

    TRAIN_PATH = "../data"
    TEST_PATH = "../data/test"
    x_train, y_train = loadEmbeddingsAndTags(TRAIN_PATH)
    x_test, y_test = loadEmbeddingsAndTags(TEST_PATH)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    print(y_train.shape)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    EMBEDDING_DIM = 100
    SENTENCE_LEN = 60
    CLASS_NUM = 14
    BATCH_SIZE = 80

    bl_stm_model = createModel(EMBEDDING_DIM, SENTENCE_LEN, CLASS_NUM)

    history = bl_stm_model.fit(x_train, y_train,
                               batch_size=BATCH_SIZE,
                               epochs=2,
                               validation_split=0.1,
                               verbose=2)

    bl_stm_model.save('../model/my_model5.h5')

    bl_stm_model.summary()

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()