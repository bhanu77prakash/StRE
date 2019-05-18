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
import aspell

import random
from empath import Empath

lexicon = Empath()
#spell = SpellChecker()
spill = aspell.Speller()

# In[2]:

stop_words = set(stopwords.words('english'))
st = StanfordNERTagger('stanford-ner/all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')

file = open(sys.argv[1], "r")
lines = file.read()
lines = lines.split("####SCORE###")
train_data = [lines[x] for x in range(len(lines)) if x%2 == 0]
y_labels = [lines[x] for x in range(len(lines)) if x%2 != 0]
y_labels = [float(x.strip()) for x in y_labels]
train_data = train_data[:-1]

final = []
for i in range(len(y_labels)):
    final.append((train_data[i], y_labels[i]))
random.shuffle(final)
train_data = []
y_labels = []
test_data = []
y_test = []
for i in range(0,len(final)-int(0.2*len(final))):
    train_data.append(final[i][0])
    y_labels.append(final[i][1])
for i in range(len(final)-int(0.2*len(final)), len(final)):
    test_data.append(final[i][0])
    y_test.append(final[i][1])

for i in range(len(y_labels)):
    if(y_labels[i]<0):
        y_labels[i] = -1
    else:
        y_labels[i] = 1

for i in range(len(y_test)):
    if(y_test[i]<0):
        y_test[i] = -1
    else:
        y_test[i] = 1


# In[6]:


check = [i.split("#$#$#")[0].strip() for i in train_data]


# In[8]:


tag_set = ['NOUN', '.', 'ADP', 'VERB', 'ADJ', 'NUM', 'CONJ', 'DET', 'PRON', 'ADV', 'PRT', 'X']
final_train_data = [] # Contains the feature vector no_oov words, POS tag counts i.e total 15 features.
for i in tqdm(range(len(check))):
    if(check[i]!=''):
        text = nltk.word_tokenize(check[i])
        senti_feat = lexicon.analyze(text, categories=['violence','swearing_terms', 'sexual', 'irritability', 'confusion', 'anonymity','emotional','lust', 'anger','ugliness','terrorism','pain', 'negative_emotion','messaging','disappointment','positive_emotion'])
        posTagged = pos_tag(text)
        simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
        counts = Counter([i[1] for i in simplifiedTags])
        words_check = [j.lower() for j in re.split('[^A-Za-z0-9]',check[i])]
        words_check = [re.sub('[^A-Za-z0-9]+', '', j) for j in words_check]
#         misspelled = set(words_check)&set(miss_1)
#         misspelled = spell.unknown(words_check)
        misspelled = [w for w in words_check if spill.check(w) == False]
        lis = [len(misspelled)]
        for j in tag_set:
            lis+=[counts[j]]
        for j in senti_feat:
            lis+=[senti_feat[j]]
        lis+=[y_labels[i]]
        final_train_data.append(lis)

# In[ ]:


train = []
for i in final_train_data:
    train.append(i[:-1])

train = np.array(train)
labels = []
for i in final_train_data:
    labels.append(i[-1])

labels = np.array(labels)

scaler = StandardScaler()
scaler.fit(train)
new_train = scaler.transform(train)


# In[ ]:


check = [i.split("#$#$#")[0].strip() for i in test_data]


tag_set = ['NOUN', '.', 'ADP', 'VERB', 'ADJ', 'NUM', 'CONJ', 'DET', 'PRON', 'ADV', 'PRT', 'X']
final_test_data = [] # Contains the feature vector no_oov words, POS tag counts i.e total 15 features.
for i in tqdm(range(len(check))):
    if(check[i]!=''):
        text = nltk.word_tokenize(check[i])
        senti_feat = lexicon.analyze(text, categories=['violence','swearing_terms', 'sexual', 'irritability', 'confusion', 'anonymity','emotional','lust', 'anger','ugliness','terrorism','pain', 'negative_emotion','messaging','disappointment','positive_emotion'])
        posTagged = pos_tag(text)
        simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
        counts = Counter([i[1] for i in simplifiedTags])
        words_check = [j.lower() for j in re.split('[^A-Za-z0-9]',check[i])]
        words_check = [re.sub('[^A-Za-z0-9]+', '', j) for j in words_check]
#         misspelled = set(words_check)&set(miss_1)
#         misspelled = spell.unknown(words_check)
        misspelled = [w for w in words_check if spill.check(w) == False]
        lis = [len(misspelled)]
        for j in tag_set:
            lis+=[counts[j]]
        for j in senti_feat:
            lis+=[senti_feat[j]]
        lis+=[y_test[i]]
        final_test_data.append(lis)


# In[ ]:


test = []
for i in final_test_data:
    test.append(i[:-1])

test = np.array(test)
truth = []
for i in final_test_data:
    truth.append(i[-1])

truth = np.array(truth)

scaler = StandardScaler()
scaler.fit(test)
new_test = scaler.transform(test)


# In[ ]:

new_data = np.zeros((len(new_train)+len(new_test), len(new_train[0])), dtype=float)
new_labels = np.zeros((len(labels)+len(truth)), dtype=float)
count = 0
for i in range(len(new_train)):
	new_data[count] = new_train[i]
	new_labels[count] = labels[i]
	count+=1

for i in range(len(new_test)):
	new_data[count] = new_test[i]
	new_labels[count] = truth[i]
	count+=1

print(sys.argv[1].split("/")[-1], len(final))
np.save("features.npy", new_data)
np.save("labels.npy", new_labels)
