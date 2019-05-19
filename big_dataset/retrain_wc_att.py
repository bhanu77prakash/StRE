'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
np.warnings.filterwarnings('ignore')

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import keras
sys.stderr = stderr

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
import sklearn
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
parser.add_argument('--epochs', type=int, default=15,
                       help='number of epochs')
parser.add_argument('--model', type=str, default='',
                       help='trained model')
parser.add_argument("--folder", type=str, default='./', help="folder containing dictionaries")
args = parser.parse_args()

filename = args.file
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
test_data = []
vali_data = []
y_test = []
vali_test = []

for i in range(0,len(final)-int(0.8*len(final))):
    train_data.append(final[i][0])
    y_labels.append(final[i][1])    


for i in range(len(final)-int(0.8*len(final)), len(final)-int(0.7*len(final))):
    vali_data.append(final[i][0])
    vali_test.append(final[i][1])


for i in range(len(final)-int(0.7*len(final)), len(final)):
    test_data.append(final[i][0])
    y_test.append(final[i][1])

    
for i in range(len(y_labels)):
    if(y_labels[i]<0):
        y_labels[i] = 0
    else:
        y_labels[i] = 1

        
for i in range(len(y_test)):
    if(y_test[i]<0):
        y_test[i] = 0
    else:
        y_test[i] = 1

for i in range(len(vali_test)):
    if(vali_test[i]<0):
        vali_test[i] = 0
    else:
        vali_test[i] = 1



max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 450
batch_size = 50
word_maxlen = 150
epochs = args.epochs
word_embed_dim = 100 
# print('Loading data...')

lens = []
for i in train_data:
    lens.append(len(i))

for i in vali_data:
    lens.append(len(i))

for i in test_data:
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

for i in vali_data:
    lens.append(len(i.strip().split()))

for i in test_data:
    lens.append(len(i.strip().split()))

lens_set = sorted(list(set(lens)))
counts = [lens.count(lens_set[0])]
for i in range(1, len(lens_set)):
    counts.append(lens.count(lens_set[i])+counts[-1])

for i in range(len(counts)):
    if counts[i] >= 0.95 * counts[-1]:
        word_maxlen = lens_set[i]
        break

embeddings_path = "glove.6B.100d-char.txt"
embeddings_dim = 100

text = ""
for i in train_data:
    text+=i

for i in test_data:
    text+=i

for i in vali_data:
    text+=i

# text = open('magic_cards.txt').read()
# print('corpus length:', len(text))

chars = sorted(list(set(text)))
# print('total chars:', len(chars))

char_indices = np.load(args.folder+'char_indices.npy').item()
indices_char = np.load(args.folder+'indices_char.npy').item()

X_train = np.zeros((len(train_data), maxlen), dtype=np.int)
X_vali = np.zeros((len(vali_data), maxlen), dtype=np.int)
Y_train = np.zeros(len(train_data), dtype=int)
X_test = np.zeros((len(test_data), maxlen), dtype=np.int)
Y_test = np.zeros(len(test_data), dtype=int)
V_test = np.zeros(len(vali_data), dtype=int)

for i, sentence in enumerate(train_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_train[i, t] = char_indices[char]
    Y_train[i] = y_labels[i]

for i, sentence in enumerate(test_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_test[i, t] = char_indices[char]
    Y_test[i] = y_test[i]

for i, sentence in enumerate(vali_data):
    for t, char in enumerate(sentence):
        if(t>=maxlen):
            break
        X_vali[i, t] = char_indices[char]
    V_test[i] = vali_test[i]

embedding_matrix = np.load(args.folder+'char_embedding_matrix.npy')
# ===============================================
# =           Loading word embeddings           =
# ===============================================

with open(args.folder+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences_train = tokenizer.texts_to_sequences(train_data)
sequences_test = tokenizer.texts_to_sequences(test_data)
sequences_vali = tokenizer.texts_to_sequences(vali_data)

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
vocab_size = len(tokenizer.word_index) + 1

word_train = sequence.pad_sequences(sequences_train, maxlen=word_maxlen)
word_vali = sequence.pad_sequences(sequences_vali, maxlen=word_maxlen)
word_test = sequence.pad_sequences(sequences_test, maxlen=word_maxlen)


word_embed_matrix = np.load(args.folder+'word_embedding_matrix.npy')

# ======  End of Loading word embeddings  =======

# =========================================
# =           attention section           =
# =========================================

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

# ======  End of attention section  =======


# =========================================
# =           Without Attention           =
# =========================================

e = Embedding(len(char_indices.keys()), embeddings_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
e_word = Embedding(vocab_size, word_embed_dim, weights=[word_embed_matrix], input_length=word_maxlen, trainable=False)

sentence_input = Input(shape=(maxlen, ), dtype='int32')
embedded_sequences = e(sentence_input)

char_lstm = Bidirectional(LSTM(64, return_sequences=True, trainable=True))(embedded_sequences)
char_att = AttLayer(embeddings_dim)(char_lstm)

word_input = Input(shape=(word_maxlen, ), dtype='int32')
word_embed_sequences = e_word(word_input)

word_lstm = Bidirectional(LSTM(64, return_sequences=True, trainable=True))(word_embed_sequences)
word_att = AttLayer(embeddings_dim)(word_lstm)
merge_one = concatenate([char_att, word_att])

# lstm_out = Bidirectional(LSTM(20))
out1 = Dense(256, activation='relu')(merge_one)
out8 = Dropout(0.5)(out1)
out2 = Dense(64, activation='relu')(out8)
out3 = Dropout(0.5)(out2)
out4 = Dense(16, activation='relu')(out3)
out5 = Dropout(0.5)(out4)
out6 = Dense(4, activation='relu')(out5)
out7 = Dense(1, activation='sigmoid')(out6)

model = Model(inputs = [sentence_input, word_input], outputs = out7)

# for l in range(len(model.layers)-8): 
#     model.layers[l].trainable=False
sgd = Adam(lr=0.001, decay=1e-4)
model.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model1 = load_model(args.model, custom_objects= {'AttLayer': AttLayer})
temp_weights = [layer.get_weights() for layer in model1.layers]

for i in range(len(temp_weights)):
    model.layers[i].set_weights(temp_weights[i])

# for l in range(len(model.layers)-8): 
#     model.layers[l].trainable=False

# model.summary()
# ======  End of Without Attention  =======

# try using different optimizers and different optimizer configs
# print('Train...')
	
model.fit([X_train, word_train], Y_train, batch_size=batch_size, epochs=epochs,  validation_data=[[X_vali, word_vali], V_test], verbose =1)
model.save('ret_model_'+filename.strip().split("/")[-1].split(".")[0]+'_epochs_'+str(epochs)+'.h5')

out = model.predict([X_test, word_test])

count_1 = 0
count_0 = 0

pred_1 = []
for i in out:
    if(i[0] >= 0.50):
        pred_1.append([1])
    else:
        pred_1.append([0])

y_test_1 = []
for i in Y_test:
	y_test_1.append([int(i)])

true_1 = 0
true_0 = 0
false_1 = 0
false_0 = 0
for i in range(len(Y_test)):
	if((Y_test[i] == 1 and out[i][0] >= 0.50) ):
		true_1 += 1
	elif(Y_test[i] == 1 and out[i][0] < 0.50):
		false_1 += 1
	elif(Y_test[i] == 0 and out[i][0] < 0.50):
		true_0 += 1
	elif(Y_test[i] == 0 and out[i][0] >= 0.50):
		false_0 += 1
	
# 
# print(count_0, count_1)

sorted_labels = [0, 1]
# print(metrics.flat_classification_report(y_test_1, pred_1, labels=sorted_labels, digits=3))
wt_f1 = sklearn.metrics.f1_score(y_test_1, pred_1, labels=sorted_labels, pos_label=1, average='weighted')
# print(count_false, count_true)

predicts = []
for i in out:
    predicts.append(i[0])
# print("AUCROC: ", roc_auc_score(Y_test, predicts))
# print(count_false, count_true)
# print("AUPRC: "+str(average_precision_score(Y_test, predicts, pos_label=1)))

print(args.file.split("/")[-1], len(final), wt_f1, roc_auc_score(Y_test, predicts), average_precision_score(Y_test, predicts, pos_label=1))