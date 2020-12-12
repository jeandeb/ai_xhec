
import os
import sys
import csv
import pickle
import argparse
import warnings
import json
warnings.filterwarnings('ignore')
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from key_generator.key_generator import generate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from utils import *

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Create Results csv for a sklearn method.')

    #MODEL
    parser.add_argument('--data', default='./data/train_data.csv',  help='Path to data csv', type=str)

    parser.add_argument('--loss', default='mean_squared_error',  help='Loss type', type=str)
    parser.add_argument('--bs',       help='Size of the batches.', default=100, type=int)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=100000)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=50000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs/')
    parser.add_argument('--two-layers',  help='', default=False, type=bool)
    parser.add_argument('--early-stop-patience',help='Epoch patience for the EarlyStopping Callback', default=100, type=int)

    return parser.parse_args(args)




def train(args=None) :


    key = generate().get_key()

    model_config = {
        'key': key,
        'bs': args.bs,
        'data': args.data,
        'lr': args.lr,
        'epochs': args.epochs,
        'steps': args.steps,
        'loss': args.loss,
        'two': args.two_layers
    }

    config_path = "model_configs/" + key + ".json"
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile)

    print("model_config ", model_config)

    #Normalisation of features.
    x_train, y_train = load_data(args.data)
    x_train = x_train.astype(np.float)
    y_train = y_train.astype(np.float)

    print("x_train.shape[0] ", x_train.shape[0])

    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=args.early_stop_patience,
        min_delta=0.001,
        restore_best_weights=True
    ))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            'models',
            '{name}.h5'.format(name=key) 
        ),
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min'
    ))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard_dir + "/" + key ))

    #Model building
    model_sequence = []
    if args.two_layers: model_sequence.append(layers.Dense(units=int(x_train.shape[0]/2)))

    model_sequence.append(layers.Dense(units=1))
    model = tf.keras.Sequential(model_sequence)

    input_shape = [None, x_train[0].shape[0]]
    model.build(input_shape)
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=args.lr),
        loss=args.loss
    )

    history = model.fit(
        x_train, 
        y_train,
        epochs=args.epochs,
        verbose=1,
        # steps_per_epoch=args.steps,
        batch_size=args.bs,
        validation_split = 0.1, 
        callbacks=callbacks
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()



    # with open('models/models_score.csv','a') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow([key, score])


    # model.fit(x_train,y_train)
    # pickle.dump(model, open("./models/"+ name+ ".pk", 'wb'))

if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)

    train(args)