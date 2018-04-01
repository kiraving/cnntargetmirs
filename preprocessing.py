# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:56:47 2018

@author: kira
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import permutation
maxlen = 76

def encoding(miseq,taseq,alinl,reverse=True):
    miseq = miseq.upper()
    taseq = taseq.upper()
    miseq = miseq.replace(' ','-')
    taseq = taseq.replace(' ','-')
    #print(miseq)
    #print(taseq)
    # let's differentiate possible bindings by their biological importances
    encode = {'AU':2,'UA':2,'AT':2,'TA':2,'CG':3,'GC':3,'UG':1,'GU':1,'TG':1,'GT':1,'AC':0,'CA':0,'AG':0,
              'GA':0,'CU':0,'UC':0,'CT':0,'TC':0,'TU':0,'UT':0}
    minlen = min(len(miseq),len(taseq)) # it should cut the "tails" where's no alignment for sure
    #taseq =taseq[:minlen]
    #miseq =miseq[:minlen]
    # reverse them in order to place seed regions in the start
    score =0
    #temp_alin= np.zeros(maxlen)
    best_alin= np.zeros(maxlen)
    if reverse:
        taseq =taseq[::-1]
        miseq =miseq[::-1]
    for t in range(len(taseq)):
        subtaseq= taseq[t:]
        temp_alin= np.zeros(maxlen)
        for j in range(minlen):
            if len(subtaseq)<=j or len(miseq)<=j:
                temp_alin[j]=0
            else:
                mi = miseq[j]
                ta = subtaseq[j]
                if mi=='-' or ta=='-' or mi==ta: # in these cases there's no alignment
                    temp_alin[j]=0
                else:
                    pair = mi+ta
                    #print(encode[pair])
                    temp_alin[j] = encode[pair]
                    #print(alinl[j])
        if sum(temp_alin)>score:
            score = sum(temp_alin)
            best_alin = temp_alin
    alinl = best_alin
    return alinl

def organism_experimental(diry,organism,target_column,replace): #'Target Site' - without alignment dashes
# 'target 5-3' - with '-' for gaps
#hsa, mmu, dre
    #human, mouse, rat, cow, dog (no exp?)
    rawtxt = pd.read_csv('miRNA_targets_%s.txt'%(organism),sep='\t',engine='c',error_bad_lines=False)
    mirwalk = pd.read_csv('%s_miRWalk_3UTR.txt'%(organism),sep='\t',engine='c',error_bad_lines=False)
    mtbase= pd.read_excel('MicroRNA_Target_Sites.xlsx',sep='\t')
    mtb = mtbase[mtbase['Species (Target Gene)']=='Homo sapiens']
    mtb = mtb.drop(['Experiments','Support Type','References (PMID)'],axis=1)
    mtb['Genesymbol']=mtb['Target Gene']
    result = pd.merge(mirwalk, mtb, on=['Genesymbol', 'miRNA'])
    rawtxt['miRNA']= rawtxt['mature miRNA']
    rawtxt['mRNA']= rawtxt['Ensembl transcript ID']
    result2 = pd.merge(rawtxt, result, on=['mRNA', 'miRNA'])
    result3 = pd.merge(rawtxt, result, on=['mRNA'])
    result3 =result3.drop(['tool name','criterion 1','criterion 2','criterion 3','target start',
                       'tsrget end','binding_site','binding_probability','miRTarBase ID','Species (miRNA)',
                       'Target Gene','Target Gene (Entrez Gene ID)','Species (Target Gene)',
                        'Ensembl transcript ID','miRNA_x','miRNA_y','Genesymbol'],axis=1)
    result3 = result3.drop_duplicates()
    alignment_for_train(diry,result3,organism,target_column,replace)

def alignment_for_train(diry,result3,name,target_column,replace):
    
    dropdup = result3[['miRNA 3-5',target_column]].drop_duplicates().reset_index(drop=True)
# the data base is sorted by mirnas, so make our shuffling even more random
    halflen = int(len(dropdup['miRNA 3-5'])/2)
    mirs0 = shuffle(dropdup['miRNA 3-5'][-halflen:]).reset_index(drop=True)
    gens0 = shuffle(dropdup[target_column][:halflen]).reset_index(drop=True)
    alin = np.zeros((len(dropdup['miRNA 3-5'])*2,maxlen))
    if replace:
    # preprocess the data
        for l in range(len(mirs0)):
    #print (mirs0[l])
            alin[l] = encoding(mirs0[l].replace('-','')
                       ,gens0[l].replace('-',''),alin[l],reverse=True)
    else:
        for l in range(len(mirs0)):
    #print (mirs0[l])
            alin[l] = encoding(mirs0[l],gens0[l],alin[l],reverse=True)
    
    mirs0 = shuffle(dropdup['miRNA 3-5'][:halflen]).reset_index(drop=True)
    gens0 = shuffle(dropdup[target_column][-halflen:]).reset_index(drop=True)

    # positive examples
    mirs1 = dropdup['miRNA 3-5'].reset_index(drop=True)
    gens1 = dropdup[target_column].reset_index(drop=True)
    if replace:
        for l in range(len(mirs0)):
    #print (mirs0[l])
            alin[l+halflen] = encoding(mirs0[l].replace('-','')
                               ,gens0[l].replace('-',''),alin[l+halflen],reverse=True)
    # positive examples
        for l in range(halflen*2):
    #print (mirs1[l])
            alin[l+halflen*2] = encoding(mirs1[l].replace('-','')
                                 ,gens1[l].replace('-',''),alin[l+halflen*2],reverse=True)
    else:
        for l in range(len(mirs0)):
    #print (mirs0[l])
            alin[l+halflen] = encoding(mirs0[l],gens0[l],alin[l+halflen],reverse=True)
    # positive examples
        for l in range(halflen*2):
    #print (mirs1[l])
                alin[l+halflen*2] = encoding(mirs1[l],gens1[l],alin[l+halflen*2],reverse=True)
# prepare the data for the neural net
# for the negative examples
    y0 = np.zeros(halflen*2)
# for the positive examples
    y1 = np.ones(halflen*2)
# targe ("answer")
    y_target = np.concatenate((y0,y1)) 
    y_perm = permutation(halflen*4)
# shuffle them
    y_target = y_target[y_perm] 
# shuffle the input data in the same order
    alin = alin[y_perm]
    print("alignments' shape: ",alin.shape)
# make a proper shape for the neuralnet's input
    alin =alin.reshape(alin.shape[0],1,alin.shape[1])
    print("reshaped alignments' shape: ",alin.shape)
    y_target = y_target.reshape(y_target.shape[0],1,1)

# save the preprocessed data
    np.save(diry+'y_%s_%s_replace%s.npy'%(name,target_column.replace(' ',''),str(replace)),y_target)
    np.save(diry+'%s_%s_replace%s.npy'%(name,target_column.replace(' ',''),str(replace)),alin)
    print("Output file names: y_%s_%s_replace%s.npy"%(name,target_column.replace(' ',''),
                                                      str(replace)),
    '%s_%s_replace%s.npy'%(name,target_column.replace(' ',''),str(replace)))
    return alin, y_target
                
                
                
                
                
def encode(mir,tar,alinl,reverse):
    #print(mir,tar)
            if len(mir)<=maxlen and len(tar)<=maxlen:
            #No problems
                alinl = encoding(mir,tar,alinl,reverse=reverse)
            else:
                    #print("A problem with maxlen..")
                    score =0
                    temp_alin= np.zeros(maxlen)
                    best_alin= np.zeros(maxlen)
                    for m in range(len(mir)-maxlen):
                        for t in range(len(tar)-maxlen):
                             temp_alin=encoding(mir[m:m+maxlen],tar[t:t+maxlen],temp_alin,reverse=reverse)
                             if sum(temp_alin)>score:
                                 score = sum(temp_alin)
                                 best_alin = temp_alin
                    alinl = best_alin    
    
def encoding_initial(miseq,taseq,alinl,reverse=True):
    miseq = miseq.upper()
    taseq = taseq.upper()
    miseq = miseq.replace(' ','-')
    taseq = taseq.replace(' ','-')
    # let's differentiate possible bindings by their biological importances
    encode = {'AU':2,'UA':2,'AT':2,'TA':2,'CG':3,'GC':3,'UG':1,'GU':1,'TG':1,'GT':1,'AC':0,'CA':0,'AG':0,
              'GA':0,'CU':0,'UC':0,'CT':0,'TC':0,'TU':0,'UT':0}
    #for i in range(15):
    #alin = np.zeros((maxlen), dtype=int)
    minlen = min(len(miseq),len(taseq)) # it should cut the "tails" where's no alignment for sure
    taseq =taseq[:minlen]
    miseq =miseq[:minlen]
    # reverse them in order to place seed regions in the start
    if reverse:
        taseq =taseq[::-1]
        miseq =miseq[::-1]
    for j in range(minlen):
            #if len(taseq)>j:
                mi = miseq[j]
                ta = taseq[j]
                if mi=='-' or ta=='-' or mi==ta: # in these cases there's no alignment
                    alinl[j]=0
                else:
                    pair = mi+ta
                    #print(encode[pair])
                    alinl[j] = encode[pair]

    return alinl