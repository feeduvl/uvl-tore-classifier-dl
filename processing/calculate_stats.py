
import numpy as np
from sklearn.metrics import confusion_matrix


def calculatePresRecall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    recall = []
    precision = []
    for n in range(len(np.sum(cm, axis=1))):
        if (np.sum(cm, axis=1)[n] == 0) and (np.sum(cm, axis=0)[n] == 0) and (np.diag(cm)[n] == 0):
            recall.append(1.0)
            precision.append(1.0)
        else:
            if np.sum(cm, axis=1)[n] != 0:
                recall.append(np.diag(cm)[n] / np.sum(cm, axis=1)[n])
            else:
                recall.append(0.0)
    for n in range(len(np.sum(cm, axis=0))):
        if np.sum(cm, axis=0)[n] != 0:
            precision.append(np.diag(cm)[n] / np.sum(cm, axis=0)[n])
        else:
            precision.append(0.0)

    return recall, precision
