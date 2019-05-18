# @Author: Bhanu Prakash Reddy
# @Date:   2019-05-18T11:17:24+05:30
# @Last modified by:   Bhanu Prakash Reddy
# @Last modified time: 2019-05-18T11:29:25+05:30

# /*=============================================>>>>>
# = Code to run various ORES++ classifiers on the dataset =
# ===============================================>>>>>*/


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

# Loading generated features and labels from the data_preprocess.py file.
features = np.load("features.npy")
labels = np.load("labels.npy")

x_train, x_val, y_train, y_val = train_test_split(features, labels,test_size = .20) # Train test split

new_train = x_train
labels = y_train
new_test = x_val
truth = y_val
print(new_train.shape, new_test.shape) # Train test sizes
# In[ ]:


# code for applying smote to balance the negative class
sm = SMOTE(ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(new_train, labels)

new_train = x_train_res
labels = y_train_res
# code ends for applying smote

# print(len([i for i in labels if i == -1]), len([i for i in labels if i == 1]))

# Code for training a simple MLP classifer on the dataset.
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

# Code for training a simple RandomForestClassifier on the dataset.
def RF():
	print("RF starts")
	model1 = RandomForestClassifier()
	model1.fit(new_train, labels)
	Y = model1.predict(new_test)

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
	wt_f1 = f1_score(y_test_1, pred_1, labels=sorted_labels, pos_label=1, average='weighted')
	print(round(wt_f1,4), round(roc_auc_score(truth, Y),4), round(average_precision_score(truth, Y),4)) # Print weighted f1 score, AUCROC, AUPRC in order
	print("\n\n=============== RF ENDS==============\n\n")

# Code for training a simple Decision Tree classifer on the dataset.
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

# Code for training a simple Logistic Regression classifer on the dataset.
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

# Code for training a simple Balanced Logistic classifer on the dataset.
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

# Call the necessary functions here. By default RF is called as it is the best ORES++ classifier we have obtained.
RF()
