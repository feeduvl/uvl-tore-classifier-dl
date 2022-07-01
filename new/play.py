# import json
#
# import numpy as np
#
# from new.SentenceGetter import SentenceGetter
# from new.dataprocess import buildDataset
# from new.training_preparation import constructModel, constructAndSaveWordAndTagMaps, getWordsAndTags, getXAndy
#
# if __name__ == "__main__":
#     PATH = "../data/train"
#     SENTENCE_LENGTH = 40
#     f = open(PATH + "/anno.json")
#     data = json.load(f)
#
#     dataset = buildDataset(data)
#     getter = SentenceGetter(dataset)
#     sentences = getter.sentences
#     print("Number of sentences: ", len(sentences))
#
#     # maxlen = max([len(s) for s in sentences])
#     # print('Maximum sequence length:', maxlen)
#
#     words, tags = getWordsAndTags(dataset)
#     n_words = len(words)
#     n_tags = len(tags)
#
#     word2idx, tag2idx = constructAndSaveWordAndTagMaps(words, tags)
#     X_train, y_train = getXAndy(word2idx, tag2idx, sentences, n_words, n_tags, SENTENCE_LENGTH)
#
#     model = constructModel(n_words, n_tags, SENTENCE_LENGTH)
#     model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#     history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=2, validation_split=0.1, verbose=1)
#
#     model.save('../new_model/' + str(SENTENCE_LENGTH) + '/model02.h5')
#     model.summary()
