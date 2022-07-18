# TORE Classifier based on Deep Learning

This is a classifier which uses the deep learning algorithm Bi-LSTM to annotate natural-language datasets.

## Usage:

#### Train

- Standard dataset is in directory ```data/train```.
- Switch to directory ```processing/```.
- Run ```python3 train.py --help``` to see the full list of options.
- Run ```python3 train.py``` to train with the standard configuration.
- Word embeddings are saved the first time. This can be disabled by setting boolean parameter ```save=False```.

#### Evaluation

- Switch to directory ```processing/```.
- Run ```python3 test.py``` to get evaluation metrics for the standard model in use.
- Run ```python3 test.py --help``` to see the full list of options.