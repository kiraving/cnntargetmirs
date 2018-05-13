# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:22:49 2018

@author: kira
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from numpy.random import permutation

from keras.models import Sequential

import argparse
import _pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from defs import *
from preprocessing import *

from datetime import datetime, date, time
import os



def parse_args():
    parser = argparse.ArgumentParser(description="micro-target")
    parser.add_argument("-o", "--outdir", dest="outdir",default=False,help="Output directory for the preduced data")
    
    parser.add_argument("-prc","--prepare_custom",help = "Input a text file with 2 columns:'miRNA 5-3' 'target 3-5'; no header")
    parser.add_argument("-prc_y","--prepare_custom_y",help = "Input a text file with 1 column:'target or not'; no header")
    parser.add_argument("-prc_eval","--prepare_custom_eval",help = "Input a text file for the evaluation with 3 columns:'miRNA 5-3' 'target 3-5' 'target or not'; no header")
    
    parser.add_argument("-pr", "--prepare_mirnamap", nargs=1, help="Input: miRNAmap table.txt. Output: alignment.npy, y_target.npy.")
    parser.add_argument("--criterion",default=False, choices=['TargetScan', 'miRanda', 'RNAhybrid','MicroTar',
    'criterion13'], help="Available only for -pr _miRNAmap table.txt_") 
    parser.add_argument("-pr_org", "--prepare_organism", nargs=3, help="Input: organism [hsa, mmu, rno],target_column ['Target Site' without alignment dashes, 'target 5-3' - with '-' for gaps], replace [True,False]")
    
    parser.add_argument("-loadt","--load_traintest_2npy",nargs=2, default=['alignment.npy', 'y_target.npy'],help="The data directory. Input: alignment.npy, y_target.npy.")    
    parser.add_argument("-tr", "--train_2npy", default=False, action='store_true', help="Input: alignment.npy, y_target.npy. Train-test split, train and evaluate")
#"Train on the training set and evaluate on the test set. Input a path to one table with 2 columns with sequences")
    parser.add_argument("-m", "--model_load", default = 'conv64-maxpool1d-32-16-1-sgd-relus.h5',help="Path to the model file")
    
    parser.add_argument("-e", "--eval", nargs=2,help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predict",  help="Data to predict on")
    
    parser.add_argument("-grid", "--grid_search",nargs=2, help="The data directory. Input: alignment.npy, y_target.npy.")
    parser.add_argument("-mplot","--plot_model",default=False, action='store_true',help ="Plot the model architecture")
    
    return parser.parse_args()





if __name__ == "__main__":
    print("start..")
    args = parse_args()
    if args.outdir:
        diry = args.outdir
    else:
        # current date and time
        diry = './'+datetime.now().strftime("%d-%m-%y-%H-%M")+'/'
        print("All the data and results will be saved to ", diry)
    if not os.path.exists(diry):
        os.makedirs(diry)
    
    with open(diry+'terminal_input.txt','w') as f:
        f.write(str(args))
    #print(args)
    maxlen = 76 #56 # suits all the mirnamap tables
    
    # Prepare data from custom text files
    # Data for prediction mode. Only alignments (X without answers).
    # Y can be prepared separately or together with X. See functions below
    if args.prepare_custom:
        rawtxt = pd.read_csv(args.prepare_custom,sep=' ',engine='c',error_bad_lines=False,header=None)
        print("\n rawtxt[1][2]: \n ",rawtxt[1][2])
        alin = np.zeros((len(rawtxt),maxlen))
        for l in range(len(rawtxt)):
            mir = rawtxt[0][l]
            tar = rawtxt[1][l]
            alin[l] = encoding(mir,tar,alin[l],reverse=False)
        print("alignments' shape: ",alin.shape)
# make a proper shape for the neuralnet's input
        alin =alin.reshape(alin.shape[0],1,alin.shape[1])
        print("reshaped alignments' shape: ",alin.shape)
# save the preprocessed data
        np.save(diry+'custom_alignment.npy',alin)
        print("Output file name: custom_alignment.npy")
    # Prepare answers (Y)
    if args.prepare_custom_y:
        rawtxt = pd.read_csv(args.prepare_custom_y,sep=' ',engine='c',error_bad_lines=False,header=None)
        y_target = np.array(rawtxt)
        print(y_target)
        y_target = y_target.reshape(y_target.shape[0],1,1)
        print(y_target)
        np.save(diry+'custom_y.npy',y_target)
    # Prepare evaluation data (X and Y)
    if args.prepare_custom_eval:
        rawtxt = pd.read_csv(args.prepare_custom_eval,sep=' ',engine='c',error_bad_lines=False,header=None)
        print("\n rawtxt[1][2]: \n ",rawtxt[1][2])
        alin = np.zeros((len(rawtxt),maxlen))
        for l in range(len(rawtxt)):
            mir = rawtxt[0][l]
            tar = rawtxt[1][l]
            alin[l] = encoding(mir,tar,alin[l],reverse=False)
        y_target = np.array(rawtxt[2])
        print(y_target)
        y_target = y_target.reshape(y_target.shape[0],1,1)
        print(y_target)
        np.save(diry+'custom_y.npy',y_target)
        print("alignments' shape: ",alin.shape)
# make a proper shape for the neuralnet's input
        alin =alin.reshape(alin.shape[0],1,alin.shape[1])
        print("reshaped alignments' shape: ",alin.shape)
# save the preprocessed data
        np.save(diry+'custom_alignment.npy',alin)
        print("Output files names: custom_alignment.npy, custom_y.npy")
        
    if args.prepare_mirnamap:
        #print (type(args.prepare_2npy))
		# read the data base with other researchers' predicted (and partially confirmed) alignments
        rawtxt = pd.read_csv(args.prepare_mirnamap,sep='\t',engine='c',error_bad_lines=False)#, encoding='utf-8') #,error_bad_lines=False)
        print("args.criterion ",args.criterion)
        if args.criterion:
            tool_name= args.criterion
            if tool_name=='criterion13':
                rawtxt = rawtxt[rawtxt["criterion 1"]>=2]
                rawtxt = rawtxt[rawtxt["criterion 3"]>0]
                rawtxt = rawtxt[rawtxt["tool name"]!='MicroTar']
            else:
                rawtxt = rawtxt[rawtxt["tool name"]==tool_name]
        else:
            rawtxt = rawtxt[rawtxt["tool name"]!='MicroTar']
# show the relevant data
        #print(rawtxt[['miRNA 3-5','target 5-3']][:5])
        alin, y_target = alignment_for_train(diry,rawtxt,tool_name,'target 5-3',False)
        
    if args.prepare_organism:
        organism = args.prepare_organism[0]
        target_column = args.prepare_organism[1]
        replace = args.prepare_organism[2]
        alin, y_target = organism_experimental(diry,organism,target_column,replace)
        
    if args.train_2npy:
        # allows to train on the data that have just been prepared
        if not args.prepare_mirnamap and not args.prepare_organism:
            print ("Loading train data: ",str(args.load_traintest_2npy))
            alin = np.load(args.load_traintest_2npy[0])
            y_target = np.load(args.load_traintest_2npy[1])
            y_target = y_target.reshape(y_target.shape[0],1,1) #just for the old data sets 
        alin,y_target,alin_test,y_target_test = traintest_split(alin,y_target)
        model = train_model(diry,alin, y_target)
        #print(classification_report(y_target_test,y_pred,digits=5))    
        full_multiclass_report(diry,model,alin_test,y_target_test,   ['0', '1'], binary=True)
    else:
        # load the -m model or default
        print("Loading model: ", args.model_load)
        model = load_model(args.model_load)
        print("Loading done")
        
    if args.eval:
        x_test = np.load(args.eval[0])
        y_test = np.load(args.eval[1])
        print("Evaluate the model ",model," on ", str(args.eval))
        full_multiclass_report(diry,model,x_test,y_test,   ['0', '1'], binary=True)
        
    if args.predict: # with probabilities
        x_test = np.load(args.predict)#[-20:]
        print("Predict for ", str(args.predict))
        y_pred = model.predict(x_test)
         #y_pred = (y_pred>0.5)

        y_pred=y_pred.reshape(y_pred.shape[0])
        print(y_pred)
        np.savetxt(diry+'prediction.txt',y_pred,delimiter='\n')
        
    if args.grid_search:
        # Parameters should be defined manually in the code
        print ("Grid Search. \n Loading train data: ",str(args.grid_search))
        alin = np.load(args.grid_search[0])
        y_target = np.load(args.grid_search[1])
        alin,y_target,alin_test,y_target_test = traintest_split(alin,y_target)
#        model = 
        search_params(diry,alin, y_target,alin_test,y_target_test)
        #print(classification_report(y_target_test,y_pred,digits=5))    
#        full_multiclass_report(diry,model,alin_test,y_target_test,   ['0', '1'], binary=True)
        
    if args.plot_model:
        plot_model(model, to_file=diry+'model.png',show_shapes=True)