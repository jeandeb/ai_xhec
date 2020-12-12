
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
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



def plot_learning_curve(estimator, title, X, y, scoring, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt, test_scores




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

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    title = r"Learning Curves"
    _, test_scores = plot_learning_curve(model, title, x_train, y_train, axes=axes[:, 0], ylim=None,
                    cv=None, n_jobs=-1, scoring=scoring)

    score = np.mean(np.abs(test_scores))
    print("test_scores", test_scores)
    print("cross_validation score", score)

    name = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}".format(args.model, args.epochs, args.lr, args.ele, args.hls, args.adaptive, args.tol, args.substances, args.no_libelle, args.bs, args.min_leaf, scoring, score)
    print("name ", name)
    with open('models/models_score.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow([args.model, args.epochs, args.lr, args.ele, args.hls, args.adaptive, args.tol, args.substances, args.no_libelle, args.bs, args.min_leaf, scoring, name, test_scores, score])

    plt.savefig("figures/" + name + ".png")

    # model.fit(x_train,y_train)
    # pickle.dump(model, open("./models/"+ name+ ".pk", 'wb'))

if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)

    train(args)