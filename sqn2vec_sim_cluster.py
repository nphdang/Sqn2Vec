import numpy as np
import timeit
import datetime
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import subprocess

### represent a sequence in form of items and sequential patterns (SPs),
### learn sequence vectors using Doc2Vec (PV-DBOW) from items and patterns simultaneously
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
def mine_SPs(file_seq, minSup, gap, file_seq_items_sp):
    subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsymsp {}".
                   format(file_seq, minSup, gap, file_seq_items_sp))

# load sequences in form of both items and SPs and their labels
def load_seq_items_SPs(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
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

print("### sqn2vec_sim_cluster, data: {}, minSup={}, gap={}, dim={} ###".format(data_name, minSup, gap, dim))
# mine SPs and associate each sequence with a set of items and SPs
in_seq = path + "/{}.txt".format(data_name)
out_seq_items_sp = path + "/{}_seq_items_sp_{}_{}.txt".format(data_name, minSup, gap)
mine_SPs(in_seq, minSup, gap, out_seq_items_sp)
# load sequences in the form of both items and SPs
data_path = path + "/{}_seq_items_sp_{}_{}.txt".format(data_name, minSup, gap)
data_X, data_y = load_seq_items_SPs(data_path)
# get true number of clusters
n_cluster = len(np.unique(data_y))
print("n_cluster: {}".format(n_cluster))
# assign a sequence id to each sequence
data_sen_X = assign_sequence_id(data_X)

all_mi_km, all_nmi_km = [], []
for run in range(n_run):
    print("run={}".format(run))
    # learn sequence vectors using Doc2Vec (PV-DBOW) from both items and SPs
    d2v_dbow = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
    d2v_dbow.build_vocab(data_sen_X)
    d2v_dbow.train(data_sen_X, total_examples=d2v_dbow.corpus_count, epochs=d2v_dbow.iter)
    data_vec = [d2v_dbow.docvecs[idx] for idx in range(len(data_sen_X))]
    del d2v_dbow  # delete unneeded model memory

    # clustering data
    print("### Algorithm: K-means ###")
    km = MiniBatchKMeans(n_clusters=n_cluster)
    km.fit(data_vec)
    mi = np.round(mutual_info_score(data_y, km.labels_), 4)
    nmi = np.round(normalized_mutual_info_score(data_y, km.labels_), 4)
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
