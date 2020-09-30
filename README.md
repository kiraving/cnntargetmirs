# CNNtargetMirs
Here is my bachelor's thesis about pairing microRNAs with the corresponding target mRNAs.
The task is to make a classifier using deep learning which predicts whether the input gene is a target for the input microRNA or not.

Can Deep Learning simplify the process of discovering new microRNA-target pairs, and how uncomplicated can be such algorithm?

Sure, it can, and an Artificial Neural Net with an ordinary architecture can achive good results!

This project was provided by the laboratory of Genomics and Bioinformatics at the Kurchatov Institute (this institute is affiliated with my Department of Nano-, Bio-, Information Technology and Cognitive Science at the Moscow Institute of Physics and Technology).

What is already done: the dataset was obtained from Targetscan (http://www.targetscan.org/vert_71/, the database itself was downloaded from MiRNAMap ftp://mirnamap.mbc.nctu.edu.tw/miRNAMap2/miRNA_Targets/Homo_sapiens/), a tool with the same purpose as mine, my set consists of microRNAs and mRNA sequences that include the respective target sites of 43-56 nucleotides and start where the seed-region starts. For negative examples, I choose an array of microRNAs and an array of randomly shuffled mRNAs. So my algorithm is trained on alignments and outputs a classification result: one or zero. For preprocessing I adjust values from 0 to 3 to binding pairs according to their biological significance: {'AU':2,'UA':2,'CG':3,'GC':3,'UG':1,'GU':1} and 0 for all the rest combinations.

Initially, the task was to match microRNAs which are about 18-24 nucleotides with target genes of an arbitrary size of about 1,000 to 100,000 nucleotides. The database was provided by the laboratory. Some previous work can be found in the folder “previous_attempts”
