


import os
import numpy as np
import pandas as pd


def convert_to_float( array ):
  list = []
  for i, elem in enumerate(array):
    list.append(np.array(elem.replace("[", "").replace("]", "").split(",")).astype(float))
  return list


def load_data(data_file, train=True, substances=False, libelle=True):

    datapath = "./data/"
    dataf = os.path.join(datapath, data_file)
    data = pd.read_csv(dataf, index_col=0)
    data = data.convert_dtypes()
    # print("data.columns ", data.columns)

    list_var = data.columns if not train else data.columns.drop("prix")

    print("list_var", list_var)
    print("small_list_var", list_var[1:-1])
    x_data = np.array( data[list_var[1:-1]])

    if libelle: 
      converted_libelle = convert_to_float(data[list_var[0]])
      x_data = np.concatenate(( converted_libelle, x_data ), axis=1)

    
    if substances: 
      converted_substances = convert_to_float(data[list_var[-1]])
      x_data = np.concatenate(( x_data, converted_substances ), axis=1)
      
    print("X DATA SHAPE: ", x_data.shape)
    y_data = None
    if train: y_data = data.prix

    return x_data, y_data