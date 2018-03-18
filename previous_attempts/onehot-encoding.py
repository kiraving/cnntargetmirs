import numpy as np

def parse_seq(a,maxlen):
    encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'U':[0,0,0,1],'N':[0,0,0,0]}
    out = np.zeros([maxlen,4])
    for idx, nt in enumerate(a):
        out[idx]=encode[nt]
    return out
out_mir = []
out_gene = []
maxlen_mir =28
fi = 'control10ktail'
len10k_gene = 0
print fi
for l in open(fi, 'r'):
    (a, b) = l.strip().split("\t")

    if len(b)<7000:
        out_mir.append(parse_seq(a,maxlen_mir))
        out_gene.append(parse_seq(b,7000))
        len10k_gene+=1

print len10k_gene

out_mir=np.array(out_mir)
np.save('out_mir_control.npy', out_mir)

out_gene = np.array(out_gene)
np.save('out_gene_control.npy', out_gene)

out_mir = []
out_gene = []
maxlen_mir =28
fi = 'input10ktail'
len10k_gene = 0
print fi
for l in open(fi, 'r'):
    (a, b) = l.strip().split("\t")

    if len(b)<7000:
        out_mir.append(parse_seq(a,maxlen_mir))
        out_gene.append(parse_seq(b,7000))
        len10k_gene+=1

print len10k_gene

out_mir=np.array(out_mir)
np.save('out_mir_input.npy', out_mir)

out_gene = np.array(out_gene)
np.save('out_gene_input.npy', out_gene)
