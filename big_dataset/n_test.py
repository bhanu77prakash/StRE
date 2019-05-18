'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''


from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from numpy import *
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU
from keras.layers import Concatenate, Input, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
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
from sklearn.metrics import precision_score, recall_score, roc_curve, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='',
                       help='file containing input.txt')
parser.add_argument('--epochs', type=int, default=5,
                       help='number of epochs')
parser.add_argument('--model', type=str, default='',
                       help='trained model')
parser.add_argument("--folder", type=str, default='./', help="folder containing dictionaries")
args = parser.parse_args()


# filename = args.file
# file = open(filename, "r")
# lines = file.read()
# lines = lines.split("####SCORE###")
# train_data = [lines[x] for x in range(len(lines)) if x%2 == 0]
# y_labels = [lines[x] for x in range(len(lines)) if x%2 != 0]
# y_labels = [float(x.strip()) for x in y_labels]
# train_data = train_data[:-1]



class AttLayer(Layer):
    def __init__(self, attention_dim=100,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])




model = load_model(args.model, custom_objects= {'AttLayer': AttLayer})

dimensions = []
for layer in model.layers:
	dimensions+=layer.get_output_at(0).get_shape().as_list()

# final = []
# for i in range(len(y_labels)):
#     final.append((train_data[i], y_labels[i]))


# random.shuffle(final)

# train_data = []
# y_labels = []

# for i in range(0,len(final)):
#     train_data.append(final[i][0])
#     y_labels.append(final[i][1])    

    
# for i in range(len(y_labels)):
#     if(y_labels[i]<0):
#         y_labels[i] = 0
#     else:
#         y_labels[i] = 1

# max_features = 20000
# # cut texts after this number of words
# # (among top max_features most common words)
# maxlen = 450
# batch_size = 50
# word_maxlen = 150
# epochs = 5

# # print('Loading data...')


# maxlen = dimensions[1]
# word_maxlen = dimensions[3]
# char_indices = np.load(args.folder+'char_indices.npy').item()
# indices_char = np.load(args.folder+'indices_char.npy').item()

# X_train = np.zeros((len(train_data), maxlen), dtype=np.int)
# Y_train = np.zeros(len(train_data), dtype=int)

# for i, sentence in enumerate(train_data):
#     for t, char in enumerate(sentence):
#         if(t>=maxlen):
#             break
#         X_train[i, t] = char_indices[char]
#     Y_train[i] = y_labels[i]

# # ===============================================
# # =           Loading word embeddings           =
# # ===============================================

# with open(args.folder+'tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# sequences_train = tokenizer.texts_to_sequences(train_data)

# word_index = tokenizer.word_index
# # print('Found %s unique tokens.' % len(word_index))
# vocab_size = len(tokenizer.word_index) + 1

# word_train = sequence.pad_sequences(sequences_train, maxlen=word_maxlen)

# ======  End of Loading word embeddings  =======

test_dict = np.load('./dict_section_2/merged_section2_0.2.scores.txt_test_dict.npy').item()
X_train = test_dict['char'] 
word_train = test_dict['word']
Y_train = test_dict['truth']


out = model.predict([X_train, word_train])

count_1 = 0
count_0 = 0

pred_1 = []
for i in out:
    if(i[0] >= 0.50):
        pred_1.append([1])
    else:
        pred_1.append([0])

y_test_1 = []
for i in Y_train:
	y_test_1.append([int(i)])

true_1 = 0
true_0 = 0
false_1 = 0
false_0 = 0
for i in range(len(Y_train)):
	if((Y_train[i] == 1 and out[i][0] >= 0.50) ):
		true_1 += 1
	elif(Y_train[i] == 1 and out[i][0] < 0.50):
		false_1 += 1
	elif(Y_train[i] == 0 and out[i][0] < 0.50):
		true_0 += 1
	elif(Y_train[i] == 0 and out[i][0] >= 0.50):
		false_0 += 1
	

# print("Class 1: True - "+str(true_1)+" False - "+str(false_1))
# print("Class 0: True - "+str(true_0)+" False - "+str(false_0))

sorted_labels = [0, 1]

# print("Model tested on "+str(filename.split('/')[-1].split('.')[0])+"\n")
print(metrics.flat_classification_report(
	    y_test_1, pred_1, labels=sorted_labels, digits=3
	))


predicts = []
for i in out:
    predicts.append(i[0])
print("AUCROC: ", roc_auc_score(Y_train, predicts))

# fpr, tpr, thresholds = metrics.roc_curve(Y_train, predicts, pos_label=1)
# print(metrics.auc(fpr, tpr))

print("AUPRC: "+str(average_precision_score(Y_train, predicts, pos_label=1)))