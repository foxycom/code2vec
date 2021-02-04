import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


def load(file):
    np.set_printoptions(precision=3, suppress=True)
    data = pd.read_csv(file)
    labels = np.array(data.pop('label')).astype('float32')
    features = np.array(data.copy())
    return features, labels


if __name__ == '__main__':
    load("data.txt")