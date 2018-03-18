import numpy as np
def parse_seq1D(a,maxlen):
    encode = {'A':1,'C':2,'G':3,'T':4,'U':5,'N':0}
    out = np.zeros([maxlen,4])
    for idx, nt in enumerate(a):
        out[idx]=encode[nt]
    return out

out_mir = []
out_gene = []
maxlen_mir =28
maxlen_gene = 3000
fi = 'mir-gene-seqs'
count_gene = 0
print fi
for l in open(fi, 'r'):
    if count_gene<100000:
    	(a, b) = l.strip().split(" ")

    	if len(b)<maxlen_gene:
        	out_mir.append(parse_seq1D(a,maxlen_mir))
        	out_gene.append(parse_seq1D(b,maxlen_gene))
        	count_gene+=1

print count_gene

out_mir=np.array(out_mir) #[:100000])
np.save('numbers_mir_mydb100k.npy', out_mir)

out_gene = np.array(out_gene) #[:100000])
np.save('numbers_gene_mydb100k.npy', out_gene)

#np.save('numbers_mir_mydb_after100k.npy', out_mir[100000:])
#np.save('numbers_gene_mydb_after100k.npy', out_gene[100000:])

#out_mir = []
#out_gene = []
#maxlen_mir =28
#fi = 'input10ktail'
#len10k_gene = 0
#print fi
#for l in open(fi, 'r'):
    #(a, b) = l.strip().split("\t")

    #if len(b)<7000:
    #    out_mir.append(parse_seq1D(a,maxlen_mir))
     #   out_gene.append(parse_seq1D(b,7000))
      #  len10k_gene+=1

#print len10k_gene

#out_mir=np.array(out_mir)
#np.save('numbers_mir_input10ktail.npy', out_mir)

#out_gene = np.array(out_gene)
#np.save('numbers_gene_input10ktail.npy', out_gene)

