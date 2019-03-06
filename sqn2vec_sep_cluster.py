import numpy as np
import timeit
import datetime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import subprocess

### represent a sequence in form of items and sequential patterns (SPs),
### learn sequence vectors using Doc2Vec (PV-DBOW) from items and SPs separately
### take average of two sequence vectors
### use K-means as clustering

### variables ###
data_name = "webkb"
path = "./data/" + data_name
minSup = 0.03
gap = 4 # 0: any gap or >0: use gap constraint
dim = 128
n_run = 10

### functions ###
# mine SPs from sequences
def mine_SPs(file_seq, minSup, gap, file_seq_sp):
    subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsp {}".
                   format(file_seq, minSup, gap, file_seq_sp))

# load sequences in form of items and their labels
def load_seq_items(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
                labels.append(label)
                sequences.append(content.rstrip().split(" "))
    return sequences, labels

# load sequences in form of SPs and their labels
def load_seq_SPs(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            labels.append(label)
            sequences.append(content.rstrip().split(" "))
    return sequences, labels

# create a sequence id to each sequence
def assign_sequence_id(sequences):
    sequences_with_ids = []
    for idx, val in enumerate(sequences):
        sequence_id = "s_{}".format(idx)
        sequences_with_ids.append(TaggedDocument(val, [sequence_id]))
    return sequences_with_ids

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

print("### sqn2vec_sep_cluster, data: {}, minSup={}, gap={}, dim={} ###".format(data_name, minSup, gap, dim))
# mine SPs and associate each sequence with a set of SPs
in_seq = path + "/{}.txt".format(data_name)
out_seq_sp = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
mine_SPs(in_seq, minSup, gap, out_seq_sp)
# load sequences in the form of items
data_path = path + "/" + data_name + ".txt"
data_i_X, data_i_y = load_seq_items(data_path)
# get true number of clusters
n_cluster = len(np.unique(data_i_y))
print("n_cluster: {}".format(n_cluster))
# assign a sequence id to each sequence
data_seq_i = assign_sequence_id(data_i_X)
# load data in the form of patterns
data_path = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
data_p_X, data_p_y = load_seq_SPs(data_path)
# assign a sequence id to each sequence
data_seq_p = assign_sequence_id(data_p_X)

all_mi_km, all_nmi_km = [], []
for run in range(n_run):
    print("run={}".format(run))
    # learn sequence vectors using Doc2Vec (PV-DBOW) from items
    d2v_i = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
    d2v_i.build_vocab(data_seq_i)
    d2v_i.train(data_seq_i, total_examples=d2v_i.corpus_count, epochs=d2v_i.iter)
    data_i_vec = [d2v_i.docvecs[idx] for idx in range(len(data_seq_i))]
    del d2v_i  # delete unneeded model memory
    # learn sequence vectors using Doc2Vec (PV-DBOW) from SPs
    d2v_p = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
    d2v_p.build_vocab(data_seq_p)
    d2v_p.train(data_seq_p, total_examples=d2v_p.corpus_count, epochs=d2v_p.iter)
    data_p_vec = [d2v_p.docvecs[idx] for idx in range(len(data_seq_p))]
    del d2v_p  # delete unneeded model memory
    # take average of sequence vectors
    data_i_vec = np.array(data_i_vec).reshape(len(data_i_vec), dim)
    data_p_vec = np.array(data_p_vec).reshape(len(data_p_vec), dim)
    data_vec = (data_i_vec + data_p_vec) / 2

    # clustering data
    print("### Algorithm: K-means ###")
    km = MiniBatchKMeans(n_clusters=n_cluster)
    km.fit(data_vec)
    mi = np.round(mutual_info_score(data_p_y, km.labels_), 4)
    nmi = np.round(normalized_mutual_info_score(data_p_y, km.labels_), 4)
    all_mi_km.append(mi)
    all_nmi_km.append(nmi)
    # obtain clustering scores
    print("mutual-information: {}".format(mi))
    print("normalized mutual-information: {}".format(nmi))

print("avg mutual-information: {} ({})".format(np.round(np.average(all_mi_km), 4), np.round(np.std(all_mi_km), 3)))
print("avg normalized mutual-information: {} ({})".format(np.round(np.average(all_nmi_km), 4),
                                                          np.round(np.std(all_nmi_km), 3)))

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))
