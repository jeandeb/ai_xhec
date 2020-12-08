import csv
import numpy as np


substances = open('./data/substances.csv') 
substances = list(csv.reader(substances))


substance_list = []
for i, substance in enumerate(substances[1:]):
    if substance[1] not in substance_list:
        substance_list.append(substance[1])
    
print(len(substance_list))


aggregated_substances = [substances[0]]
aggr_pos = 0

for i, substance in enumerate(substances[1:]):
    index = substance_list.index(substance[1])

    if substance[0] != aggregated_substances[aggr_pos][0]: 

        encoding = list(np.zeros((len(substance_list))))
        encoding[index] = 1
        aggregated_substances.append([substance[0], encoding])

        aggr_pos += 1
    else:
        # print("APPEND")
        # print("aggregated_substances[aggr_pos]", aggregated_substances[aggr_pos])
        aggregated_substances[aggr_pos][1][index] = 1


    # print(aggregated_substances)
    # print("--")


with open('./data/aggregated_substances.csv', 'w') as csvfile:
    csvfile_writer = csv.writer(csvfile)
    csvfile_writer.writerows(aggregated_substances) 



