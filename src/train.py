import json

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers import TimeDistributed
import tensorflow as tf
import numpy as np


def loadEmbeddingsAndTags():
    f = open("../data/embeddings.json")
    embeddings = json.load(f)
    f = open("../data/tags.json")
    tags = json.load(f)
    return embeddings, tags


def createModel(emb_dim, sen_len, class_num):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape=(sen_len, emb_dim))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(TimeDistributed(Dense(class_num, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    return model


if __name__ == "__main__":
    all_embeddings, all_tags = loadEmbeddingsAndTags()

    all_tags = to_categorical(all_tags)

    all_embeddings = np.asarray(all_embeddings)
    all_tags = np.asarray(all_tags)

    EMBEDDING_DIM = 100
    SENTENCE_LEN = 80
    CLASS_NUM = 14
    BATCH_SIZE = 773

    bl_stm_model = createModel(EMBEDDING_DIM, SENTENCE_LEN, CLASS_NUM)

    checkpoint_path = "../model/training_1/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = bl_stm_model.fit(all_embeddings, all_tags,
                               batch_size=BATCH_SIZE,
                               epochs=7,
                               validation_data=(all_embeddings, all_tags),
                               verbose=2,
                               callbacks=[cp_callback])

    bl_stm_model.summary()
