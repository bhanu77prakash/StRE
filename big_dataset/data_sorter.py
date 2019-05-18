'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from numpy import *
import os, sys
import random
from tqdm import tqdm
import nltk
import pickle
import argparse
import os
import operator
from operator import itemgetter

def datapts_counter(filename):
    file = open(filename, "r")
    lines = file.read()
    lines = lines.split("####SCORE###")
    train_data = [lines[x] for x in range(len(lines)) if x%2 == 0]
    y_labels = [lines[x] for x in range(len(lines)) if x%2 != 0]
    y_labels = [float(x.strip()) for x in y_labels]
    train_data = train_data[:-1]
    return len(train_data)



parser = argparse.ArgumentParser()

parser.add_argument('--folder', type=str, default='',
                       help='folder containing input.txt')
parser.add_argument('--thresh', type=float, default=0.9,
                       help='threshold of data points')
parser.add_argument('--list_file', type=bool, default=True, help='outputs the list of files merged in a file')

parser.add_argument('--merge', type=str, default="", help='Files to be merged explicitly')
args = parser.parse_args()

train_data = []
y_labels = []
test_data = []
vali_data = []
y_test = []
vali_test = []

list_files = []
# filename = args.file
count_list = []
sum_counts = 0
if(args.folder != ''):
    for filename in os.listdir(args.folder):
        filename = os.path.join(args.folder, filename)
        count_list.append((filename, datapts_counter(filename)))
        sum_counts+=count_list[-1][1]

count_list.sort(key=operator.itemgetter(1),reverse=True)
cum_sum = 0

for i in count_list:
    if(cum_sum > args.thresh*sum_counts):
        break
    list_files.append(i[0])
    cum_sum += i[1]


if (args.merge!=""):
	merge_files = open(args.merge,"r")
	merge_files = merge_files.readlines()
	if(merge_files[-1] == ''):
		merge_files=merge_files[:-1]
	for i in range(len(merge_files)):
		merge_files[i] = os.path.join(args.folder, merge_files[i]).strip()
	for filename in os.listdir(args.folder.split("scores")[0]+"train_scores"):
		filename = os.path.join(args.folder.split("scores")[0]+"train_scores", filename).strip()
		merge_files.append(filename)
	list_files = merge_files
comp_data = []

print("Considering the "+str(args.thresh)+" percent data from " + str(len(list_files)) + " files")

final_file = args.folder.strip().split("/")[-2]
if(final_file == 'scores'):
    final_file = args.folder.strip().split("/")[-3]

write_file = open("merged_"+final_file+"_"+str(args.thresh)+".scores.txt", 'w')
if(args.list_file):
	list_file = open("merged_"+final_file+"_"+str(args.thresh)+".list.txt", 'w')
for filename in list_files:
    file = open(filename, "r")
    lines = file.read()
    write_file.write(lines)
    if(args.list_file):
    	list_file.write(filename.strip().split("/")[-1].split(".")[0]+"\n")

write_file.close()
if(args.list_file):
	list_file.close()
