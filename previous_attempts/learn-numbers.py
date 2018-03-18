
# coding: utf-8

# In[ ]:
#KERAS_BACKEND=theano python -c "from keras import backend"
import numpy as np
from numpy.random import permutation
import keras
print keras.backend.backend()
from keras.utils import np_utils
from keras.models import Model, Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.optimizers import SGD, RMSprop
from keras import backend
from keras.layers import Input, merge
from keras.layers.core import Dense #, Lambda, Reshape
#from keras.layers.convolutional import Conv1D
from keras.callbacks import CSVLogger
from keras.models import Model
print "# Importing Done"

# Loading preprocessed data for training of our neural net
genes_1 = np.load('numbers_gene_input10ktail.npy') # 100 000 target genes from the head of "input_for_neuronet" file
mirs_1 = np.load('numbers_mir_input10ktail.npy') # 100 000 microRNAs for these targets
genes_0 = np.load('numbers_gene_control10ktail.npy') # 100 000 non-target genes from the head of "control_for_neuronet"
mirs_0 = np.load('numbers_mir_control10ktail.npy') # their microRNAs
print "# Loading Done"

genes = np.concatenate((genes_1, genes_0)) # merging target and non-target genes
targets_1 = np.empty(genes_1.shape[0]) # making of neural net's "positive answers" (for targets) 
targets_1.fill(1) 
targets_0 = np.zeros(genes_1.shape[0]) # making of "negative answers"
targets = np.concatenate((targets_1, targets_0)) # merging answers together into a single vector according to the genes' statuses (targets or non-targets)
mirs = np.concatenate((mirs_1, mirs_0)) # merging microRNAs

perm = permutation(len(genes)) # making a set of numbers in a random order for further permutation
genes = genes[perm] # here genes are reordered so we no longer maintain the exact sequence of negative answers following the positives ones
mirs = mirs[perm] # micrornas and answers get the same order
targets = targets[perm]

# Cutting of 20% of the data set for test
train_size = int(0.8*len(genes))
print train_size, type(train_size), len(genes)
genes_t = genes[train_size:-1]
mirs_t = mirs[train_size:-1]
targets_t = targets[train_size:-1]
# the rest is for training
genes = genes[0:train_size]
mirs = mirs[0:train_size]
targets = targets[0:train_size]

print "# Sampling Done"

# Neural net's architecture
dense1 = 64 # number of neurons in the first hidden layer
dense2 = 32 # in the second one
batch_size = 128 # during training our pairs of mirnas are devided into small batches (this method speeds up the training process)

# making a part of neural net for genes 
genes_branch = Sequential()
genes_branch.add(Dense(dense1, input_dim=genes_1.shape[1])) # first layer of neurons
genes_branch.add(Activation('relu')) # ReLu is their activation function
genes_branch.add(Dropout(0.5)) # a half of the least effective connections between neurons are dropped 
genes_branch.add(Dense(dense2)) # the second layer
genes_branch.add(Activation('relu'))
genes_branch.add(Dropout(0.5))

# the same architecture for microRNAs
mirs_branch = Sequential()
mirs_branch.add(Dense(dense1, input_dim=genes_1.shape[1]))
mirs_branch.add(Activation('relu'))
mirs_branch.add(Dropout(0.5))
mirs_branch.add(Dense(dense2))
mirs_branch.add(Activation('relu'))
mirs_branch.add(Dropout(0.5))

# merging two neural nets into a single one
R_Q_D = Merge([mirs_branch, genes_branch], mode = 'concat') 
model_Rs = Sequential()
model_Rs.add(R_Q_D)
model_Rs.add(Dense(1)) # adding a so-called decision neuron 
model_Rs.add(Activation('sigmoid')) # Sigmoid activation function

csv_logger = CSVLogger('learn-numbers.log') # saving the log

model_Rs.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics=['accuracy','recall','precision']) # compiling the neural net
print "# Compilation Done"

# Starting training process
model_Rs.fit([mirs, genes], targets, # data and answers
             batch_size=batch_size,
              nb_epoch=20, callbacks=[csv_logger],
              validation_split=0.1) # cross-validation on 10% 

print "# Fitting Done"
model_Rs.save('learn-numbers-100k_model.h5') # Saving the model (to extract the weight coefficents in case we need them in the future)
print "# Saving Done"

# testing the neural net's performance
print model_Rs.evaluate([mirs_t, genes_t], targets_t, batch_size=batch_size, callbacks=[csv_logger])
print "# Testing Done"

