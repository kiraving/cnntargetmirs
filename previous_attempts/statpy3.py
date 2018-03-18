
# coding: utf-8

# In[1]:

import numpy as np
import math

mir_grams = { 'A':0, 'C':1, 'U':2, 'T':2, 'N': 2, 'G': 3 }
gen_grams = { 'A':0, 'C':1, 'U':2, 'T':2, 'N': 2, 'G': 3 }
#{ 'A':2, 'C':3, 'U':0, 'T':0, 'N': 0, 'G': 1 }
N_GRAM = 7
SEEDLEN = 16
MISMATCH = 2
# перестановок N_GRAM! / MISMATCH! (N_GRAM - MISMATCH)! = 7! / 2!(7-2)! = 21
# и умножить на замещение букв * 3**MISMATCH
# + 1 оригинальная строка н-грама
# умножить на колво разбиений на н-грамы = SEEDLEN -N_GRAM +1

# gram_vector = np.zeros((1 + 3*N_GRAM)*(maxlen_mir - N_GRAM +1))
arr_len = 1+ (3**MISMATCH)*math.factorial(N_GRAM)/math.factorial(MISMATCH)/math.factorial(N_GRAM-MISMATCH)
vector_len = arr_len*(SEEDLEN - N_GRAM +1)
arr_len, vector_len


# In[5]:

def mir_gramto4system(ngram, grams):
    out = 0
    i=0
    # ngram.reverse()
    for c in ngram:
        i+=1
        out += grams[c]*(4**(N_GRAM - i))
        #print c, out
    return out
def gen_gramto4system(ngram, grams):
    out = 0
    i=0
    for c in ngram:
        i+=1
        out += grams[c]*(4**(N_GRAM - i))
        #print c, out
    return out
def gramto4system(ngram, grams):
    out = 0
    i=0
    for c in ngram:
        i+=1
        out += grams[c]*(4**(N_GRAM - i))
        #print c, out
    return out
def numbto4system(numb):
    out = 0
    i=0
    for c in numb:
        i+=1
        out += c*(4**(N_GRAM - i))
        #print c, out
    return out
def gramtotempvec(ngram, grams):
    temparr = np.zeros([arr_len, N_GRAM])
    tempvec = np.zeros(arr_len)
    i=0
    count = 0
    for c in ngram:
        for j in xrange(0, arr_len):
            temparr[j,i] = grams[c]
        i+=1
    #print "original"
    #print temparr[0]
    for i in xrange(0, N_GRAM-1):
        #print i
        for k in xrange(i+1, N_GRAM):
            #print k
            index = count*9
            if temparr[-1,i]==0: temparr[index:index+9,i] = [1,1,1,2,2,2,3,3,3]
            if temparr[-1,i]==1: temparr[index:index+9,i] = [0,0,0,2,2,2,3,3,3]
            if temparr[-1,i]==2: temparr[index:index+9,i] = [0,0,0,1,1,1,3,3,3]
            if temparr[-1,i]==3: temparr[index:(index+9),i] = [0,0,0,1,1,1,2,2,2]
            if temparr[-1,k]==0: temparr[index:(index+9),k] = [1,2,3,1,2,3,1,2,3]
            if temparr[-1,k]==1: temparr[index:index+9,k] = [0,2,3,0,2,3,0,2,3]
            if temparr[-1,k]==2: temparr[index:index+9,k] = [0,1,3,0,1,3,0,1,3]
            if temparr[-1,k]==3: temparr[index:index+9,k] = [0,1,2,0,1,2,0,1,2]
            count+=1
            # print count
    #print "temparr : "
    #print temparr[54:-1]
    for j in xrange(0, arr_len):
        #print temparr[j].astype(int)
        tempvec[j] = numbto4system(temparr[j].astype(int))
        #print "tempvec %(je)s" %{"je" : j}
        #print tempvec[j]
    return tempvec

def parsemir(a1):
    a = a1[::-1]
    gram_vector = np.zeros(vector_len)
    mirvec = np.zeros(vector_len)
    gram4sys = 0
    for i in xrange(0, SEEDLEN -N_GRAM +1): # SEEDLEN -N_GRAM +1 = 2
        gram_vector[i*arr_len: (i+1)*arr_len] = gramtotempvec(a[i:i+7],mir_grams)
        #print "gram_vector"
        #print gram_vector[i*arr_len: (i+1)*arr_len]
        #print len(gramtotempvec(a[i:i+7],N_GRAM))
        #print i*(1 + 3*N_GRAM), (i+1)*(1 + 3*N_GRAM)
        #mirvec[((i+1)*(1 + 3*N_GRAM) - 1)] +=1
        gram4sys = gramto4system(a[i:i+7],mir_grams)
        if gram4sys in gram_vector:
            mirvec[int(np.where(gram_vector==gram4sys)[0][0])] +=1
            #print "mirvec[int"
            #print mirvec[int(np.where(gram_vector==gram4sys)[0][0])]
            #print "[int where]"
            #print int(np.where(gram_vector==gram4sys)[0][0])
        # print gram_vector[i*(1 + 3*N_GRAM): (i+1)*(1 + 3*N_GRAM)]
    #print "mirvec"
    #print mirvec
    return gram_vector, mirvec
        
def parsegene(b, gramvector):
    gram4sys = 0
    genevec = np.zeros(vector_len)
    for i in xrange(0, len(b)-6):
        gram4sys = gramto4system(b[i:i+7], gen_grams)
        if gram4sys in gramvector:
            #print "np.where(gramvector==gram4sys)"
            #print np.where(gramvector==gram4sys)
            genevec[int(np.where(gramvector==gram4sys)[0][0])] +=1
    #print "genevec"
    #print genevec
    return genevec  


# In[3]:

def parse_file(fi):
    out_mir = []
    out_gene = []
    for l in open(fi, 'r'):
    #if count<3:
        (a, b) = l.strip().split("\t")
        #print a
        [gramvector, mirvector] = parsemir(a)
        out_gene.append(parsegene(b, gramvector))
        out_mir.append(mirvector)
    mir=np.array(out_mir)
    np.save('%(file)s-7gram_mir_seed16_v3.npy' %{"file": fi} , mir)
    gene = np.array(out_gene)
    np.save('%(file)s-7gram_gene_seed16_v3.npy' %{"file": fi} , gene)


# In[85]:

fi = 'input_for_neuronet'
parse_file(fi)


