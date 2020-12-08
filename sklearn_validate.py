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

    x_test, _ = load_data("new_test.csv", train=False)
    test = pd.read_csv("./data/new_test.csv", index_col=0)
    test = test.convert_dtypes()

    model = pk.load(open(args.model, "rb"))
    results = model.predict(x_test)

    results_new_shape = np.empty((results.shape[0] + 1, 2), dtype='object')
    results_new_shape[0,:] = ['id','prix']
    results_new_shape[1:,1] = results
    results_new_shape[1:,0] = test.index


    with open(os.path.join('results', "{0}.csv".format(os.path.basename(args.model[:-3]))), 'w') as csvfile:
        results_writer = csv.writer(csvfile)
        results_writer.writerows(results_new_shape) 