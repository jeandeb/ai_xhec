

# import csv
# import keras
# import numpy as np

# max_value = 1000

# testcsvfile = open('./data/new_test.csv') 
# testfilelist = list(csv.reader(testcsvfile))
# x_test = np.asarray(testfilelist).astype(float)
# print(x_test.shape)

# traincsvfile = open('./data/new_train.csv') 
# trainfilelist = list(csv.reader(traincsvfile))

# train_set = np.asarray(trainfilelist).astype(float)

# x_train = train_set[:,:-1]
# y_train = train_set[:,-1]/max_value

# entry_layer_size = x_train[0].shape



# model = keras.models.Sequential()
# model.add(keras.Input(shape=(entry_layer_size[0],)))
# model.add(keras.layers.Dense(32, activation='relu')) 
# model.add(keras.layers.Dense(1))
# model.output_shape

import os
import sys
import csv
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from utils import *

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Create Results csv for a sklearn method.')

    #MODEL
    parser.add_argument('--data', default='./data/train_data.csv',  help='Path to data csv', type=str)
    parser.add_argument('--epochs', default=50000,  help='Number of epochs.', type=int)
    parser.add_argument('--bs', default=0,  help='Batch size.', type=int)
    parser.add_argument('--lr', default=1e-6,  help='Number of epochs.', type=float)
    parser.add_argument('--hls', default=100,  help='hidden_layer_sizes.', type=int)
    parser.add_argument('--ele', default=1000,  help='hidden_layer_sizes.', type=int)
    parser.add_argument('--adaptive', default=False,  help='adaptive lr.', type=bool)
    parser.add_argument('--tol', default=0,  help='minimum increase for adaptive lr.', type=float)
    parser.add_argument('--substances', default=False,  help='', type=bool)
    parser.add_argument('--no_libelle', default=False,  help='', type=bool)
    parser.add_argument('--model', default="mlp" ,  help='adaptive lr.', type=str)
    parser.add_argument('--min_leaf', default=20,  help='', type=int)

    return parser.parse_args(args)


def train(args=None) :
    scoring = 'neg_mean_squared_error'

    x_train, y_train = load_data(args.data)

    model = None
    #TRAINING MODEL
    if args.model == "mlp":
        model = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto' if args.bs <= 0 else args.bs, beta_1=0.9,
                    beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=(args.hls,), learning_rate='adaptive' if args.adaptive else 'constant',
                    learning_rate_init=args.lr, max_fun=15000, max_iter=args.epochs,
                    momentum=0.9, n_iter_no_change=args.ele, nesterovs_momentum=True,
                    power_t=0.5, random_state=None, shuffle=True, solver='sgd' if args.adaptive else 'adam',
                    tol=args.tol, validation_fraction=0.1, verbose=1,
                    warm_start=False)
                    
    elif args.model == "hgbr":
        model = HistGradientBoostingRegressor(
                    max_iter=args.epochs, 
                    learning_rate=args.lr,
                    verbose=1,
                    max_leaf_nodes=31,
                    min_samples_leaf=args.min_leaf)

    cross_validation = cross_validate(model, x_train, y_train, cv=5, n_jobs=-1, return_estimator=False, scoring=scoring)
    score = np.mean(np.abs(cross_validation['test_score']))
    print("cross_validation", cross_validation)
    print("cross_validation score", score)

    name = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}".format(args.model, args.epochs, args.lr, args.ele, args.hls, args.adaptive, args.tol, args.substances, args.no_libelle, args.bs, args.min_leaf, scoring, score)
    print("name ", name)
    with open('models/models_score.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow([args.model, args.epochs, args.lr, args.ele, args.hls, args.adaptive, args.tol, args.substances, args.no_libelle, args.bs, args.min_leaf, scoring, name, cross_validation['test_score'], score])


    model.fit(x_train,y_train)
    pickle.dump(model, open("./models/"+ name+ ".pk", 'wb'))

if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)

    train(args)