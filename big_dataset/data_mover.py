import os
import sys

folder = sys.argv[1]

file = open(sys.argv[2], "r")
files = file.readlines()
if(files[-1] == ''):
	files = files[:-1]
print(files[-1])


if(os.path.isdir(folder+"train_scores")==False):
	os.mkdir(folder+"train_scores")
for i in files:
	os.system("mv "+folder+"scores/"+i.strip()+".scores.txt "+folder+"train_scores")

# os.system("mv "+folder+"scores " + " retrain_scores")