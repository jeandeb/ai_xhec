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




# 5: 'tx rembours': 65%
# 9: 'date declar annee': 20120101
# 10: 'date amm annee': 20120101
# 13: 'prix': 2.7
    # break


# substances_objects = substances[['substance']]
# # print("substances_objects shape", substances_objects.shape)
# encoder_substances = OneHotEncoder()
# encoder_substances.fit(substances_objects)

# substances_aggregated = pd.DataFrame()
# aggr_pos = 0

# # print("substances: \n", substances.head(200))
# # print("substances: \n", substances.shape)
# # print("substances: \n", substances.columns)

# for idx, substance_in_data_point in substances.iterrows():
#     print("--")
#     # print(substance_in_data_point['substance'])
#     # print(substance_in_data_point['id'])
#     # print(substances_aggregated['id'])

#     encoding = encoder_substances.transform([[substance_in_data_point['substance']]]).toarray()[0]
#     print("encoding", encoding[:10])
#     print("substance_in_data_point['id'] ", substance_in_data_point['id'])
#     print("substances_aggregated", substances_aggregated)
#     if substances_aggregated.shape[0] <= 0 or substance_in_data_point['id'] != substances_aggregated[aggr_pos]: 
#         print("Creating")
#         row = pd.DataFrame([list([substance_in_data_point['id']]) + list(encoding)])
#         print("[substance_in_data_point['id']] ", [substance_in_data_point['id']])
#         print("row", row)
#         substances_aggregated = substances_aggregated.append(row)
#         aggr_pos += 1
#         # break
#     else:
#         print("Adding")

#         current_encoding = substances_aggregated[aggr_pos]
#         print("current_encoding", current_encoding)
#         substances_aggregated[aggr_pos]

#     print("substances_aggregated[aggr_pos]", substances_aggregated[aggr_pos])

#     # print(aggregated_substances)




#-----------------------------------
# substances = open('./data/substances.csv') 
# substances = list(csv.reader(substances))


# substance_list = []
# for i, substance in enumerate(substances[1:]):
#     if substance[1] not in substance_list:
#         substance_list.append(substance[1])
    
# print(len(substance_list))


# aggregated_substances = [substances[0]]
# aggr_pos = 0

# for i, substance in enumerate(substances[1:]):
#     index = substance_list.index(substance[1])

#     if substance[0] != aggregated_substances[aggr_pos][0]: 

#         encoding = list(np.zeros((len(substance_list))))
#         encoding[index] = 1
#         aggregated_substances.append([substance[0], encoding])

#         aggr_pos += 1
#     else:
#         # print("APPEND")
#         # print("aggregated_substances[aggr_pos]", aggregated_substances[aggr_pos])
#         aggregated_substances[aggr_pos][1][index] = 1


#     # print(aggregated_substances)
#     # print("--")


# with open('./data/aggregated_substances.csv', 'w') as csvfile:
#     csvfile_writer = csv.writer(csvfile)
#     csvfile_writer.writerows(aggregated_substances) 





# for row in list_csv[1:]:
#     # print(', '.join(row))
#     # print(row[1], row[2], row[3], row[4], row[5])
#     new_line = []
#     new_line.append(row[0])

#     found = False
#     new_row = row[1].replace('\"','')
#     for elem in libelle_fe_reader[1:]:
#         # print(row[1],  elem[0])
#         new_elem = elem[0].replace('\"', '')
#         # print(new_elem)
#         # print(new_row, new_elem)
#         if new_row == new_elem: 
#             new_elem = []
#             for num in elem[1:]:
#                 new_elem.append(float(num))
#             new_line.append(new_elem)
#             found = True
#             break

#     if not found :
#         array = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
#         new_line.append(array)

# print(mean_libelle)

# 5: 'tx rembours': 65%
# 9: 'date declar annee': 20120101
# 10: 'date amm annee': 20120101
# 13: 'prix': 2.7


# libelle_fe = open('./data/libelle_fe.csv')
# libelle_fe_reader = list(csv.reader(libelle_fe))
# testcsvfile = open('./data/test.csv')
# testfilereader = list(csv.reader(testcsvfile))
# traincsvfile = open('./data/train.csv')
# trainfilereader = list(csv.reader(traincsvfile))

# substances = open('./data/aggregated_substances.csv')
# substances_reader = list(csv.reader(substances))

# positions = list(np.asarray(substances_reader)[:,0])

# def get_classes_dict( list1, list2, row_num ):

#     assignmement = {}
#     tmp_list = list1 + list2
#     for elem in tmp_list:
#         if elem[row_num] not in assignmement:
#             num = len(assignmement.keys())
#             assignmement[elem[row_num]] = num

#     return assignmement

# row_1_dict = get_classes_dict( testfilereader, trainfilereader, 1 )
# row_2_dict = get_classes_dict( testfilereader, trainfilereader, 2 )
# row_3_dict = get_classes_dict( testfilereader, trainfilereader, 3 )
# row_4_dict = get_classes_dict( testfilereader, trainfilereader, 4 )

# row_6_dict = get_classes_dict( testfilereader, trainfilereader, 6 )
# row_7_dict = get_classes_dict( testfilereader, trainfilereader, 7 )
# row_8_dict = get_classes_dict( testfilereader, trainfilereader, 8 )

# row_11_dict = get_classes_dict( testfilereader, trainfilereader, 11 )
# row_12_dict = get_classes_dict( testfilereader, trainfilereader, 12 )




# def get_new_data_format( list_csv, train=False ): 
#     new_data = []
    # for row in list_csv[1:]:
    #     # print(', '.join(row))
    #     # print(row[1], row[2], row[3], row[4], row[5])
    #     new_line = []
    #     new_line.append(row[0])

    #     found = False
    #     new_row = row[1].replace('\"','')
    #     for elem in libelle_fe_reader[1:]:
    #         # print(row[1],  elem[0])
    #         new_elem = elem[0].replace('\"', '')
    #         # print(new_elem)
    #         # print(new_row, new_elem)
    #         if new_row == new_elem: 
    #             new_elem = []
    #             for num in elem[1:]:
    #                 new_elem.append(float(num))
    #             new_line.append(new_elem)
    #             found = True
    #             break

    #     if not found :
    #         array = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    #         new_line.append(array)

#         # print(row[0])
#         # new_line.append(row_2_dict[row[1]])
#         new_line.append(float(row_2_dict[row[2]]))
#         new_line.append(float(row_3_dict[row[3]]))
#         new_line.append(float(row_4_dict[row[4]]))
#         new_line.append(float(row[5][:-1]))
#         new_line.append(float(row_6_dict[row[6]]))
#         new_line.append(float(row_7_dict[row[7]]))
#         new_line.append(float(row_8_dict[row[8]]))
#         new_line.append(float(row[9]))
#         new_line.append(float(row[10]))
#         new_line.append(float(row_11_dict[row[11]]))
#         new_line.append(float(row_12_dict[row[12]]))


#         if len(row) >= 14:
#             new_line.append(float(row[13]))

#            #Get substances
#         index = positions.index(row[0])
#         new_line.append(substances_reader[index][1])


#         new_data.append(new_line)
        
#     return new_data

# new_test = [testfilereader[0] + ["substances"]] + get_new_data_format(testfilereader)
# new_train = [trainfilereader[0] + ["substances"]] + get_new_data_format(trainfilereader)
# # val = [trainfilereader[0] + ["substances"]] + get_new_data_format(trainfilereader)[:int(len(trainfilereader)/20)]

# with open('./data/new_test_oh.csv', 'w') as csvfile:
#     new_test_writer = csv.writer(csvfile)
#     new_test_writer.writerows(new_test) 

# with open('./data/new_train_oh.csv', 'w') as csvfile:
#     new_train_writer = csv.writer(csvfile)
#     new_train_writer.writerows(new_train) 

# # with open('./data/validation_oh.csv', 'w') as csvfile:
# #     val_writer = csv.writer(csvfile)
# #     val_writer.writerows(new_train[:int(len(new_train)/20)]) 