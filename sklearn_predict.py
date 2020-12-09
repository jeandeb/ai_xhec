#GETTING RESULTS
import csv
import sys
import argparse
import pandas as pd
import pickle as pk
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from utils import *


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Create Results csv for a sklearn method.')

    #MODEL
    parser.add_argument('model',              help='Path to the model.')
    return parser.parse_args(args)


if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)

    data = pd.read_csv("./data/test_data.csv", index_col=0).to_numpy()
    x_test = data[:,1:]

    model = pk.load(open(args.model, "rb"))
    results = model.predict(x_test)

    results_new_shape = np.empty((results.shape[0] + 1, 2), dtype='object')
    results_new_shape[0,:] = ['id','prix']
    results_new_shape[1:,1] = results
    results_new_shape[1:,0] = data[:,0]


    with open(os.path.join('results', "{0}.csv".format(os.path.basename(args.model[:-3]))), 'w') as csvfile:
        results_writer = csv.writer(csvfile)
        results_writer.writerows(results_new_shape) 