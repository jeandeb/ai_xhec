


import os
import numpy as np
import pandas as pd


def convert_to_float( array ):
  list = []
  for i, elem in enumerate(array):
    list.append(np.array(elem.replace("[", "").replace("]", "").split(",")).astype(float))
  return list


def load_data(data_file, train=True):

    data = pd.read_csv(data_file, index_col=0).to_numpy()
    print("data\n", data[:3])

    x_data = data[:,1:-1]
    print("x_data\n", x_data[:3])
    print("X x_data.shape: ", x_data.shape)

    y_data = None
    if train: 
      y_data = data[:,-1]
      print("y_data\n", y_data[:3])

    return x_data, y_data