# TORE Classifier based on Deep Learning

This is a classifier which uses the deep learning algorithm Bi-LSTM to annotate natural-language datasets.

## Project Structure

### Current model

The current model is always saved at ```model/model.h5```. The prediction algorithm automatically uses this path to find the model.

### New models

If is does not exist yet, create directory ```models/{SENTENCE_LENGTH}/```, as this is where the model is saved after training.

## Usage:

#### Train

- Standard dataset is in directory ```data/train```.
- Switch to directory ```processing/```.
- Run ```python3 train.py --help``` to see the full list of options.
- Run ```python3 train.py``` to train with the standard configuration.
- Word embeddings are saved the first time. If a different dataset is used, the word embeddings must be deleted. The word embeddings are saved at ```/data/generated/```.

#### Evaluation

- Switch to directory ```processing/```.
- Run ```python3 test.py``` to get evaluation metrics for the standard model in use.
- Run ```python3 test.py --help``` to see the full list of options.
- Word embeddings are saved the first time. If a different dataset is used, the word embeddings must be deleted. The word embeddings are saved at ```/data/generated/test/```.