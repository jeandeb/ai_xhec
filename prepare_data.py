#DATA PREPARATION
import csv
import numpy as np

libelle_fe = open('./data/libelle_fe.csv')
libelle_fe_reader = list(csv.reader(libelle_fe))
testcsvfile = open('./data/test.csv')
testfilereader = list(csv.reader(testcsvfile))
traincsvfile = open('./data/train.csv')
trainfilereader = list(csv.reader(traincsvfile))

substances = open('./data/aggregated_substances.csv')
substances_reader = list(csv.reader(substances))

positions = list(np.asarray(substances_reader)[:,0])

def get_classes_dict( list1, list2, row_num ):

    assignmement = {}
    tmp_list = list1 + list2
    for elem in tmp_list:
        if elem[row_num] not in assignmement:
            num = len(assignmement.keys())
            assignmement[elem[row_num]] = num

    return assignmement

row_1_dict = get_classes_dict( testfilereader, trainfilereader, 1 )
row_2_dict = get_classes_dict( testfilereader, trainfilereader, 2 )
row_3_dict = get_classes_dict( testfilereader, trainfilereader, 3 )
row_4_dict = get_classes_dict( testfilereader, trainfilereader, 4 )

row_6_dict = get_classes_dict( testfilereader, trainfilereader, 6 )
row_7_dict = get_classes_dict( testfilereader, trainfilereader, 7 )
row_8_dict = get_classes_dict( testfilereader, trainfilereader, 8 )

row_11_dict = get_classes_dict( testfilereader, trainfilereader, 11 )
row_12_dict = get_classes_dict( testfilereader, trainfilereader, 12 )




def get_new_data_format( list_csv, train=False ): 
    new_data = []
    for row in list_csv[1:]:
        # print(', '.join(row))
        # print(row[1], row[2], row[3], row[4], row[5])
        new_line = []
        new_line.append(row[0])

        found = False
        new_row = row[1].replace('\"','')
        for elem in libelle_fe_reader[1:]:
            # print(row[1],  elem[0])
            new_elem = elem[0].replace('\"', '')
            # print(new_elem)
            # print(new_row, new_elem)
            if new_row == new_elem: 
                new_elem = []
                for num in elem[1:]:
                    new_elem.append(float(num))
                new_line.append(new_elem)
                found = True
                break

        if not found :
            array = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            new_line.append(array)

        # print(row[0])
        # new_line.append(row_2_dict[row[1]])
        new_line.append(float(row_2_dict[row[2]]))
        new_line.append(float(row_3_dict[row[3]]))
        new_line.append(float(row_4_dict[row[4]]))
        new_line.append(float(row[5][:-1]))
        new_line.append(float(row_6_dict[row[6]]))
        new_line.append(float(row_7_dict[row[7]]))
        new_line.append(float(row_8_dict[row[8]]))
        new_line.append(float(row[9]))
        new_line.append(float(row[10]))
        new_line.append(float(row_11_dict[row[11]]))
        new_line.append(float(row_12_dict[row[12]]))


        if len(row) >= 14:
            new_line.append(float(row[13]))

           #Get substances
        index = positions.index(row[0])
        new_line.append(substances_reader[index][1])


        new_data.append(new_line)
        
    return new_data

new_test = [testfilereader[0] + ["substances"]] + get_new_data_format(testfilereader)
new_train = [trainfilereader[0] + ["substances"]] + get_new_data_format(trainfilereader)
# val = [trainfilereader[0] + ["substances"]] + get_new_data_format(trainfilereader)[:int(len(trainfilereader)/20)]

with open('./data/new_test_oh.csv', 'w') as csvfile:
    new_test_writer = csv.writer(csvfile)
    new_test_writer.writerows(new_test) 

with open('./data/new_train_oh.csv', 'w') as csvfile:
    new_train_writer = csv.writer(csvfile)
    new_train_writer.writerows(new_train) 

# with open('./data/validation_oh.csv', 'w') as csvfile:
#     val_writer = csv.writer(csvfile)
#     val_writer.writerows(new_train[:int(len(new_train)/20)]) 