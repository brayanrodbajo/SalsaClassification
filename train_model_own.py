from common import GENRES
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Activation, \
        TimeDistributed, Convolution1D, MaxPooling1D, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from optparse import OptionParser
from sys import stderr, argv
import os

SEED = 42
N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 256
BATCH_SIZE = 32
EPOCH_COUNT = 10

def train_model(data, model_path):
    x = data['x']
    y = data['y']
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.3,
            random_state=SEED)

    print('Building model...')

    n_features = x_train.shape[2]

    model = models.Sequential()
    model.add(Convolution1D(64, FILTER_LENGTH, activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(MaxPooling1D(2))
    model.add(layers.Convolution1D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling1D((2, 2)))
    model.add(layers.Convolution1D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling1D((2, 2)))
    model.add(layers.Convolution1D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling1D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--data_path', dest='data_path',
            default=os.path.join(os.path.dirname(__file__),
                'data/data.pkl'),
            help='path to the data pickle', metavar='DATA_PATH')
    parser.add_option('-m', '--model_path', dest='model_path',
            default=os.path.join(os.path.dirname(__file__),
                'models/model_salsa.h5'),
            help='path to the output model HDF5 file', metavar='MODEL_PATH')
    options, args = parser.parse_args()

    with open(options.data_path, 'rb') as f:
        data = pickle.load(f)

    train_model(data, options.model_path)
