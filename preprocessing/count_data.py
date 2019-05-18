# ===============================================================================================================
# =           Code to count the number of datapoints in a processed file or folder of processed files           =
# ===============================================================================================================




import numpy as np
import os, sys
import random
from tqdm import tqdm
import pickle
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='',
                       help="folder containing input.txt's")
parser.add_argument('--file', type=str, default='',
                       help='file containing input.txt')
parser.add_argument('--csv', type=str, default='',
						help='file to append the count of data points')
args = parser.parse_args()


# Function to count the number of datapoints in a file

def datapts_counter(filename):
	file = open(filename, "r")
	lines = file.read()
	lines = lines.split("####SCORE###")
	train_data = [lines[x] for x in range(len(lines)) if x%2 == 0]
	y_labels = [lines[x] for x in range(len(lines)) if x%2 != 0]
	y_labels = [float(x.strip()) for x in y_labels]
	train_data = train_data[:-1]
	return len(train_data)

# Given a folder, counts the number of datapoints in each file
# Given a file, counts the number of datapoints in the file

if __name__ == '__main__':
	savedir = ''
	writer = ''
	if(args.csv != ''):
		savedir = open(args.csv, 'a')
		writer = csv.writer(savedir)
	if(args.folder != ''):
		for filename in os.listdir(args.folder):
			filename = os.path.join(args.folder, filename)
			count = datapts_counter(filename)
			if(args.csv != ''):
				writer.writerow([filename.strip().split('/')[-1], str(count)])

	else:
		count = datapts_counter(filename)
		if(args.csv != ''):
			writer.writerow([filename.strip().split('/')[-1], str(count)])
			
	if(args.csv != ''):
		savedir.close()
