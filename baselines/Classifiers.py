#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.corpus import words
#from spellchecker import SpellChecker
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from collections import Counter
from nltk.tag import pos_tag, map_tag
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import aspell
from sklearn import tree
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve

# In[2]:

features = np.load("features.npy")
labels = np.load("labels.npy")

x_train, x_val, y_train, y_val = train_test_split(features, labels,test_size = .20)

new_train = x_train
labels = y_train
new_test = x_val
truth = y_val
print(new_train.shape, new_test.shape)
# In[ ]:
'''
#Code for balancing training data

count = 0
for i in range(len(labels)):
	if(labels[i] == -1):
		count+=1

list_of_ones = [i for i in range(len(labels)) if labels[i] == 1]
sampled = random.sample(list_of_ones, count)
new_train_1 = np.zeros(new_train.shape, dtype=float)
labels_1 = np.zeros(labels.shape)
count = 0
for i in range(len(labels)):
	if(labels[i] == -1):
		sampled+=[i]

for i in sampled:
	new_train_1[count] = new_train[i]
	labels_1[count] = int(labels[i])
	count+=1
new_train = new_train_1
labels= labels_1
# Code for balancing training data ends
'''


# code for applying smote

sm = SMOTE(ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(new_train, labels)

new_train = x_train_res
labels = y_train_res

# code ends for applying smote

print(len([i for i in labels if i == -1]), len([i for i in labels if i == 1]))

def train_MLP():
	print("MPL Starting")
	model = MLPClassifier(verbose=True, max_iter = 1000)
	model.fit(new_train, labels)
	Y = model.predict(new_test)
	print("MLP \n log_loss: " + str(log_loss(truth, Y)))
	print("MLP \n Accuracy: " + str(np.mean(truth == Y)))

	y_test_1 = []
	for i in truth:
	    y_test_1.append([i])
	    
	pred_1 = []
	for i in Y:
	    pred_1.append([i])

	sorted_labels = [-1, 1]
	print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))

	print(roc_auc_score(truth, Y))
	fpr, tpr, thresholds = roc_curve(truth, Y)
	print(precision_score(truth, Y), recall_score(truth, Y), fpr[0], tpr[0])
	print("\n\n ============= MLP ENDS ===============\n\n")
# model = MLPClassifier(verbose=True, max_iter = 1000)
# results = cross_validate(model, new_train, labels, return_train_score=False, scoring = ('neg_log_loss'))
# print("\n\n\n"+str(results['test_score'])+"\n\n\n")
# In[ ]:

def RF():
	# print("RF starts")
	model1 = RandomForestClassifier()
	model1.fit(new_train, labels)
	Y = model1.predict(new_test)
	# print("\nRandom \n log_loss: "+str(log_loss(truth, Y)))
	# print("\nRandom \n Accuracy: "+str(np.mean(truth == Y)))

	y_test_1 = []
	for i in truth:
	    y_test_1.append([i])
	    
	pred_1 = []
	for i in Y:
		if(i<0):
			pred_1.append([-1])
		else:
			pred_1.append([1])

	sorted_labels = [-1, 1]
	# print(metrics.flat_classification_report(
	#     y_test_1, pred_1, labels=sorted_labels, digits=3
	# ))
	wt_f1 = f1_score(y_test_1, pred_1, labels=sorted_labels, pos_label=1, average='weighted')
	print(round(wt_f1,4), round(roc_auc_score(truth, Y),4), round(average_precision_score(truth, Y),4))

	# fpr, tpr, thresholds = roc_curve(truth, Y)
	# print(precision_score(truth, Y), recall_score(truth, Y), fpr[0], tpr[0])
	# print(roc_auc_score(truth, Y))
	# print("\n\n=============== RF ENDS==============\n\n")
# In[ ]:

'''
model2 = MultinomialNB()
model2.fit(new_train, labels)
Y = model2.predict(new_test)
print("\nMNB \n log_loss: "+str(log_loss(truth, Y)))
print("\nMNB \n Accuracy: "+str(np.mean(truth == Y)))

y_test_1 = []
for i in truth:
    y_test_1.append([i])
    
pred_1 = []
for i in Y:
    pred_1.append([i])

sorted_labels = [-1, 1]
print(metrics.flat_classification_report(
    y_test_1, pred_1, labels=sorted_labels, digits=3
))
'''

def DT():
	print("DT starts")
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(new_train, labels)
	Y = clf.predict(new_test)
	print("\nDecision \n log_loss: "+str(log_loss(truth, Y)))
	print("\nDecision \n Accuracy: "+str(np.mean(truth == Y)))
	y_test_1 = []
	for i in truth:
	    y_test_1.append([i])
	    
	pred_1 = []
	for i in Y:
	    pred_1.append([i])

	sorted_labels = [-1, 1]
	print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))
	print(roc_auc_score(truth, Y))
	fpr, tpr, thresholds = roc_curve(truth, Y)
	print(precision_score(truth, Y), recall_score(truth, Y), fpr[0], tpr[0])
	print("\n\n===============DT ends============\n\n")

def  Logistic():
	print("logistic starts")
	model1 = LogisticRegression(class_weight=None)
	model1.fit(new_train, labels)
	Y = model1.predict(new_test)
	print("\nLogistic \n log_loss: "+str(log_loss(truth, Y)))
	print("\nLogistic \n Accuracy: "+str(np.mean(truth == Y)))

	y_test_1 = []
	for i in truth:
	    y_test_1.append([i])
	    
	pred_1 = []
	for i in Y:
	    pred_1.append([i])

	sorted_labels = [-1, 1]
	print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))

	print(roc_auc_score(truth, Y))
	print("\n\n===============logistic ends============\n\n")


def bal_logistic():
	print("bal_logisitc starts")
	model1 = LogisticRegression(class_weight='balanced')
	model1.fit(new_train, labels)
	Y = model1.predict(new_test)
	print("\nLogistic with balanced weights \n log_loss: "+str(log_loss(truth, Y)))
	print("\nLogistic with balanced weights \n Accuracy: "+str(np.mean(truth == Y)))

	y_test_1 = []
	for i in truth:
	    y_test_1.append([i])
	    
	pred_1 = []
	for i in Y:
	    pred_1.append([i])

	sorted_labels = [-1, 1]
	print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))

	print(roc_auc_score(truth, Y))
	print("\n\n===============bal_logistic ends============\n\n")

# train_MLP()
RF()
# DT()
# bal_logistic()
