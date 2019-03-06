# Sqn2Vec: Unsupervised Learning Sequence Embeddings via Sequential Patterns
This is the implementation of the Sqn2Vec method in the paper "Sqn2Vec: Learning Sequence Representation via Sequential Patterns with a Gap Constraint", ECML-PKDD 2018: https://link.springer.com/chapter/10.1007/978-3-030-10928-8_34

# Introduction
A sequence dataset consists of multiple sequences, and each sequence is an ordered list of discrete symbols (items). It can be seen in many real-world applications, e.g., text mining, action recognition, navigation analysis, system diagnosis, and so on.

There are two types of machine learning tasks on sequences, namely sequence classification and sequence clustering. However, to apply machine learning methods to sequences, we need to construct/learn feature vectors for sequences first.

We propose Sqn2Vec which learns feature vectors (aka embeddings or representations) for sequences. Sqn2Vec combines two important techniques in data mining and machine learning: sequential pattern mining and neural embedding learning. It first decomposes each sequence into a set of sequential patterns (SPs) and then learns an embedding for each sequence by predicting its belonging SPs. It has two different models: (1) Sqn2Vec-SEP which learns sequence embeddings from symbols and SPs separately and (2) Sqn2Vec-SIM which learns sequence embeddings from symbols and SPs simultaneously.

![Sqn2Vec: Two models](https://github.com/nphdang/Sqn2Vec/blob/master/two_models.jpg)

# Installation
1. Microsoft .NET Framework 4.0 (to run C# code to mine sequential patterns)
2. gensim 3.4 (to run Doc2Vec model) 

# How to run
- To use Sqn2Vec-SEP model, run "python sqn2vec_sep_classify.py" to learn sequence embeddings and classify sequences (note that you may need to change variables such as dataset, minimum support threshold, gap constraint, and embedding dimension in the code)
- To use Sqn2Vec-SIM model, run "python sqn2vec_sim_classify.py" to learn sequence embeddings and classify sequences
- Use "sqn2vec_sep_cluster.py" and "sqn2vec_sim_cluster.py" for the clustering task

# Tool to mine sequential patterns
- File "sp_miner.exe" can be used as a standalone tool to discover sequential patterns
- Its source code is in "sp_miner.zip"
- Its parameters are as follows:
```
        -dataset <file>
        use sequences from <file> to mine SPs        
        -minsup <float>
        set minimum support threshold in [0,1]; default is 0.5
        -gap <int>
        set gap constraint > 0; set 0 if don't use gap constraint
        -sp <file>
        save discovered SPs to <file> (optional)
        -seqsp <file>
        convert each sequence to a set of SPs and save it to <file> (optional)
        -seqsymsp <file>
        convert each sequence to a set of symbols and SPs and save it to <file> (optional)
```
# Reference
Dang Nguyen, Wei Luo, Tu Dinh Nguyen, Svetha Venkatesh, Dinh Phung (2018). Sqn2Vec: Learning Sequence Representation via Sequential Patterns with a Gap Constraint. ECML-PKDD 2018, Dublin, Ireland. Springer LNCS, 11052, 569-584
