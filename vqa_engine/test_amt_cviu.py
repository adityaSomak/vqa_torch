import csv
import os
dir = 'D:\drive2\For PhD\KR Lab\ASU_Vision\CVIU_2017_AMT'

def create_joined_dataset():
    new_ratings_dict = {}
    with open(os.path.join(dir,'new_coco_correctness.csv.tsv'), 'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        header = reader.next()
        header1 = header[2:4]
        header1.extend(header[8:-2])
        for row in reader:
            key = row[2][row[2].rfind("/")+1:]
            to_join = [row[3]]
            to_join.extend(row[8:-2])
            new_ratings_dict[key] = to_join

    print new_ratings_dict.items()[0]

    with open(os.path.join(dir,'coco_correctness.tsv'), 'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter='\t')
        header = reader.next()
        header = [h+"_old" for h in header]
        header1.extend(header[0:2])
        header1.extend(header[5:-2])
        for row in reader:
            key = row[0][row[0].rfind("/")+1:]
            try:
                to_join = [row[1]]
                to_join.extend(row[5:-2])
                new_ratings_dict[key].extend(to_join)
            except KeyError:
                pass

    print header1
    print new_ratings_dict.items()[0]
    with open(os.path.join(dir,'joined_coco_correctness.tsv'), 'w') as csvfile:
        csvfile.write("\t".join(header1) +"\n")
        for key,value in new_ratings_dict.items():
            csvfile.write(key+"\t")
            csvfile.write("\t".join(value) + "\n")

# create_joined_dataset()
new_ratings_dict = {}
with open(os.path.join(dir,'joined_coco_correctness.csv.tsv'), 'rb') as csvfile:
    reader = csv.reader(csvfile,delimiter='\t')
    header = reader.next()
    for row in reader:
        new_ratings_dict[row[0]] = row[1:]

import scipy.stats as stats

all_keys = list(new_ratings_dict.keys())
GT_new = [int(new_ratings_dict[k][1]) for k in all_keys]
GT_old = [int(new_ratings_dict[k][7]) for k in all_keys]
tau, p_value = stats.kendalltau(GT_new, GT_old)
print 'tau: %f, pvalue: %f' % (tau, p_value)

karp_new = [int(new_ratings_dict[k][2]) for k in all_keys]
karp_old = [int(new_ratings_dict[k][8]) for k in all_keys]
tau, p_value = stats.kendalltau(karp_new, karp_old)
print 'tau: %f, pvalue: %f' % (tau, p_value)

sdg1_new = [int(new_ratings_dict[k][3]) for k in all_keys]
sdg1_old = [int(new_ratings_dict[k][9]) for k in all_keys]
tau, p_value = stats.kendalltau(sdg1_new, sdg1_old)
print 'tau: %f, pvalue: %f' % (tau, p_value)