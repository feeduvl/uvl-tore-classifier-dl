import json

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical


def loadEmbeddingsAndTags(path):
    f = open(path + "/embeddings.json")
    embeddings = json.load(f)
    f = open(path + "/tags.json")
    tags = json.load(f)
    return embeddings, tags


if __name__ == "__main__":
    TEST_PATH = "../data/test"
    x_test, y_test = loadEmbeddingsAndTags(TEST_PATH)

    y_test = to_categorical(y_test)

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Recreate the exact same model, including its weights and the optimizer
    new_model = tf.keras.models.load_model('../model/my_model5.h5')

    # Show the model architecture
    new_model.summary()

    # loss, acc = new_model.evaluate(x_test, y_test, verbose=2)
    # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    i = 0
    p = new_model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)
    print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
    for w, pred in zip(x_test[i], p[0]):
        print("{:14}: {}".format(w, pred))