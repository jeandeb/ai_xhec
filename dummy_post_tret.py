

import csv

results = open('./results/sklearn_1.csv') 
results = list(csv.reader(results))

for i, result in enumerate(results[1:]):
    if float(result[1]) <= 0: results[i+1][1] = 1


with open('./results/sklean_1_new.csv', 'w') as csvfile:
    csvfile_writer = csv.writer(csvfile)
    csvfile_writer.writerows(results) 
