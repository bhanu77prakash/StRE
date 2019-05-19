# ===========================================================================================================
# =           Code that pre-processes the data and generated dictionaries that are needed further           =
# ===========================================================================================================




from __future__ import print_function
import numpy as np

from numpy import *
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU
from keras.layers import Concatenate, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os, sys
import random
from keras.optimizers import SGD, Adam
from imblearn.over_sampling import SMOTE
import sklearn_crfsuite

from sklearn.preprocessing import StandardScaler
from keras.layers.core import *
from keras.models import *
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
import pickle

# ===========================================
# =           Read the input file           =
# ===========================================

filename = sys.argv[1]
file = open(filename, "r")
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

for i in range(0,len(final)):
    train_data.append(final[i][0])
    y_labels.append(final[i][1])    

    
for i in range(len(y_labels)):
    if(y_labels[i]<0):
        y_labels[i] = 0
    else:
        y_labels[i] = 1
# ======  End of Read the input file  =======


# Hyperparameters

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 450
batch_size = 50
word_maxlen = 150
epochs = 5

print('Loading data...')

# ===============================================================================================================
# =           Defining the max length of the word and character sequences using cummulative frequency           =
# ===============================================================================================================

lens = []
for i in train_data:
    lens.append(len(i))

lens_set = sorted(list(set(lens)))
counts = [lens.count(lens_set[0])]
for i in range(1, len(lens_set)):
    counts.append(lens.count(lens_set[i])+counts[-1])

for i in range(len(counts)):
	if counts[i] >= 0.95 * counts[-1]:
		maxlen = lens_set[i]
		break

lens = []
for i in train_data:
    lens.append(len(i.strip().split()))

lens_set = sorted(list(set(lens)))
counts = [lens.count(lens_set[0])]
for i in range(1, len(lens_set)):
    counts.append(lens.count(lens_set[i])+counts[-1])

for i in range(len(counts)):
    if counts[i] >= 0.95 * counts[-1]:
        word_maxlen = lens_set[i]
        break

# ======  End of Defining the max length of the word and character sequences using cummulative frequency  =======


embeddings_path = "glove.6B.100d-char.txt"
embeddings_dim = 100

text = ""
for i in train_data:
    text+=i


# text = open('magic_cards.txt').read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Saving character dictionay and inverted character dictionary
np.save('char_indices.npy', char_indices)
np.save('indices_char.npy', indices_char)

X_train = np.zeros((len(train_data), maxlen), dtype=np.int)
Y_train = np.zeros(len(train_data), dtype=int)

# ===========================================
# =           Character embedding           =
# ===========================================

for i, sentence in enumerate(train_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_train[i, t] = char_indices[char]
    Y_train[i] = y_labels[i]



embedding_vectors = {}
with open(embeddings_path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

embedding_matrix = np.zeros((len(chars), embeddings_dim))
#embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
for char, i in char_indices.items():
    #print ("{}, {}".format(char, i))
    embedding_vector = embedding_vectors.get(char)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Saving character embedding matrix
np.save('char_embedding_matrix.npy', embedding_matrix)

# ===============================================
# =           Loading word embeddings           =
# ===============================================

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


sequences_train = tokenizer.texts_to_sequences(train_data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
vocab_size = len(tokenizer.word_index) + 1

word_train = sequence.pad_sequences(sequences_train, maxlen=word_maxlen)

word_embed_index = dict()
word_embed_dim = 100

f = open('./glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	word_embed_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(word_embed_index))
# create a weight matrix for words in training docs
word_embed_matrix = zeros((vocab_size, word_embed_dim))
for word, i in tokenizer.word_index.items():
	word_embed_vector = word_embed_index.get(word)
	if word_embed_vector is not None:
		word_embed_matrix[i] = word_embed_vector

np.save('word_embedding_matrix.npy', word_embed_matrix)
# ======  End of Loading word embeddings  =======
