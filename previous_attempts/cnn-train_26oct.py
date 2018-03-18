import numpy as np
import math
from sklearn.utils import shuffle

from os.path import join,exists
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Merge
from keras.layers.convolutional import Conv2D,MaxPooling2D,MaxPooling1D,Conv1D
from keras.optimizers import Adadelta,RMSprop

from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from random import randint
from sklearn.cross_validation import train_test_split
from keras import backend as K

mirs= np.load('numbers_mir_mydb100k.npy')
out_gene = np.load('numbers_gene_mydb100k.npy')
print mirs.shape

# Generation of 'negative' genes by simply shuffling the original genes' 
set
from numpy.random import permutation
genes_1 = out_gene
#perm1 = permutation(genes_1.shape[0])
genes_0 = shuffle(genes_1[50000:])
genes_1=genes_1[:50000]

genes = np.concatenate((genes_1, genes_0))
                        #, genes_02, genes_03
                        #, genes_04))
targets_1 = np.empty(genes_1.shape[0])
targets_1.fill(1) 
targets_0 = np.zeros(genes_1.shape[0])
targets = np.concatenate((targets_1, targets_0))
#mirs = out_mir #np.concatenate((out_mir, out_mir))
                       #, mirs_1, mirs_1
                       #, mirs_1))

perm = permutation(genes.shape[0])
genes = genes[perm]
mirs = mirs[perm]
targets = targets[perm]
# For test
train_size = int(0.8*genes_1.shape[0])
print train_size
genes_t = genes[train_size:-1]
mirs_t = mirs[train_size:-1]
targets_t = targets[train_size:-1]

genes = genes[0:train_size]
mirs = mirs[0:train_size]
targets = targets[0:train_size]

print "split done"

def run_model(X_train_g,X_train_m, Y_train, X_test_g,X_test_m, Y_test):
   
    DROPOUT = 0.5 #{{choice([0.3,0.5,0.7])}}
    csv_logger = CSVLogger('training-cnn-26oct.log') # saving the log
    

    gmodel = Sequential()
    gmodel.add(Conv1D(20,4, input_shape=(3000,4),activation='relu'))
    gmodel.add(MaxPooling1D(2))
    gmodel.add(Flatten())
    gmodel.add(Dense(32,activation='relu'))
    gmodel.add(Dropout(DROPOUT))
    
    mmodel = Sequential()
    mmodel.add(Conv1D(20,4, input_shape=(28,4),activation='relu'))
    mmodel.add(MaxPooling1D(2))
    mmodel.add(Flatten())
    mmodel.add(Dense(32,activation='relu'))
    mmodel.add(Dropout(DROPOUT))
    
    merged = Merge([gmodel, mmodel], mode = 'concat')
    modelmg = Sequential()
    modelmg.add(merged)
    #modelmg.add(Dense(32,activation='relu'))
    modelmg.add(Dropout(DROPOUT))
    modelmg.add(Dense(1))
    modelmg.add(Activation('softmax'))
    print "layers are added"

    myoptimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    mylossfunc = 'binary_crossentropy'
    modelmg.compile(loss=mylossfunc, 
optimizer=myoptimizer,metrics=['accuracy','recall','precision'])
    print "compiled"
    modelmg.fit([X_train_g,X_train_m], Y_train, batch_size=100,nb_epoch=20, callbacks=[csv_logger],
              validation_split=0.1)
    print "fitted"
    score, acc = modelmg.evaluate([X_test_g,X_test_m], Y_test)
    print "evaluated"
    print('Test accuracy:', acc,'test loss',score)

print "model defined"

run_model(genes,mirs,targets,genes_t,mirs_t,targets_t)
