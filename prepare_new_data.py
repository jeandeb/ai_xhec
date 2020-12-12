#DATA PREPARATION
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


test_data = pd.read_csv('./data/test.csv')
# print("test_data: \n", test_data.head(3))

train_data = pd.read_csv('./data/train.csv')
# print("train_data: \n", train_data.head(3))

substances = pd.read_csv('./data/substances.csv')
substances_agg = substances.groupby('id')['substance'].apply(lambda x:','.join(x)).reset_index()
train_data = pd.merge(train_data, substances_agg, how='left')
test_data = pd.merge(test_data, substances_agg, how='left')

# print("substances: \n", substances.head(3))

libelle_fe = pd.read_csv('./data/libelle_fe.csv')
libelle_fe = libelle_fe.drop_duplicates(subset='libelle')

#Simple datapoints ('tx rembours', 'date declar annee', 'date amm annee', 'prix' )
#-------
train_data_ids = train_data[['id']].to_numpy()
train_data_simple_points = train_data[['tx rembours', 'date declar annee', 'date amm annee', 'prix']].to_numpy()
train_data_simple_points[:,0] = np.array(list(map(lambda s: s.replace('%' , ''), train_data_simple_points[:,0]))).astype(np.float)

test_data_ids = test_data[['id']].to_numpy()
test_data_simple_points = test_data[['tx rembours', 'date declar annee', 'date amm annee']].to_numpy()
test_data_simple_points[:,0] = np.array(list(map(lambda s: s.replace('%' , ''), test_data_simple_points[:,0]))).astype(np.float)
# print(list(map(lambda s: s.replace('%' , ''), train_data_simple_points[:,0])))
#------------

#ONE HOT ENCODING OF TRIVIAL COLUMNS
data_point_to_one_hot_encode = {'statut': 2, 'etat commerc': 3, 'agrement col': 4, 'forme pharma': 6, 'voies admin': 7, 'statut admin': 8, 'type proc': 11, 'titulaires': 12}

train_data_objects = train_data[list(data_point_to_one_hot_encode.keys())]
test_data_objects = test_data[list(data_point_to_one_hot_encode.keys())]

all_data_objects = pd.concat([train_data_objects,test_data_objects])

encoder = OneHotEncoder()
encoder.fit(all_data_objects)

train_data_onehotlabels = encoder.transform(train_data_objects).toarray()
test_data_onehotlabels = encoder.transform(test_data_objects).toarray()
#-----------------------------------

#LIBELLE PROCESSING
mean_pd_libelle = libelle_fe.mean(axis = 0)

def get_libelles( dataframe, libelledata ):
    columns = libelledata.columns

    libelle_df = pd.DataFrame(columns=columns[1:])
    for datapoint in dataframe:
        # print("datapoint", datapoint)
        row = libelledata.loc[libelledata['libelle'] == datapoint]
        if row.shape[0] == 0: libelle_df = libelle_df.append(mean_pd_libelle, ignore_index=True)
        else: libelle_df = libelle_df.append(row)

    return libelle_df

train_data_libelles = get_libelles(train_data['libelle'], libelle_fe).to_numpy()[:,:-1]
test_data_libelles = get_libelles(test_data['libelle'], libelle_fe).to_numpy()[:,:-1]
#-----------------------------------

#SUBSTANCES PROCESSING


# substance_labels = pd.get_dummies(substances.substance)
# print(substance_labels.columns)
# print(substance_labels.head(10))
# print(substance_labels.shape)

substances_objects = substances[['substance']]
encoder_substances = OneHotEncoder()
encoder_substances.fit(substances_objects)

# print(encoder_substances.drop_idx_.shape)

# substance_encoder = OneHotEncoder()
# substance_encoder.fit([substances['substance']])
# print(encoder.transform([substances['substance']]).toarray())
# print(encoder.transform([substances['substance']]).toarray().shape)
def get_substance_codes( data ):

    codes = []
    for idx, datapoint in data.iterrows():

        datapoints_substances = np.array(datapoint['substance'].split(","))
        # label = np.array(substance_labels[datapoints_substances[0]])
        label = encoder_substances.transform([[datapoints_substances[0]]]).toarray()[0]

        # print(label.shape)
        for substance in datapoints_substances[1:]:
            # label += np.array(substance_labels[substance])
            label += encoder_substances.transform([[substance]]).toarray()[0]

        codes.append(np.array(label))

    return np.array(codes)

train_substance_codes = get_substance_codes( train_data )
test_substance_codes = get_substance_codes( test_data )
# #----------
print("train_data_onehotlabels.shape", train_data_onehotlabels.shape)
print("train_data_libelles.shape", train_data_libelles.shape)
print("train_substance_codes.shape", train_substance_codes.shape)
print("train_data_simple_points.shape", train_data_simple_points.shape)



print("test_data_onehotlabels.shape", test_data_onehotlabels.shape)
print("test_data_libelles.shape", test_data_libelles.shape)
print("test_substance_codes.shape", test_substance_codes.shape)
print("test_data_simple_points.shape", test_data_simple_points.shape)


# print("train_data_onehotlabels.shape\n", train_data_onehotlabels[:2])
# print("train_data_libelles.shape\n", train_data_libelles[:2])
# print("train_substance_codes.shape\n", train_substance_codes[2])


# print("test_data_onehotlabels.shape\n", test_data_onehotlabels[:2])
# print("test_data_libelles.shape\n", test_data_libelles[:2])
# print("test_substance_codes.shape\n", test_substance_codes[:2])


conc_train_data = np.concatenate( (train_data_ids, train_data_onehotlabels, train_data_libelles, train_substance_codes, train_data_simple_points), axis=1 )
print("conc_train_data.shape", conc_train_data.shape)
print("conc_train_data[:2]\n", conc_train_data[:2])
pd.DataFrame(conc_train_data).to_csv("./data/train_data.csv")

conc_test_data = np.concatenate( (test_data_ids, test_data_onehotlabels, test_data_libelles, test_substance_codes, test_data_simple_points), axis=1  )
print("conc_test_data.shape", conc_test_data.shape)
print("conc_test_data[:2]\n", conc_test_data[:2])
pd.DataFrame(conc_test_data).to_csv("./data/test_data.csv")

