import numpy as np
import networkx as nx
from scipy.io import loadmat
import scipy.sparse as sp
from itertools import combinations

np.random.seed(1994)

def generate_data(adj, adj_train, feats, labels, mask, shuffle=True):
    adj = adj.tocsr()
    adj_train = adj_train.tocsr()
    feats = feats.tocsr()
    zipped = list(zip(adj, adj_train, feats, labels, mask))
    while True: # this flag yields an infinite generator
        if shuffle:
            print('Shuffling data')
            np.random.shuffle(zipped)
        for data in zipped:
            a, t, f, y, m = data
            yield (a.toarray(), t.toarray(), f.toarray(), y, m)


def batch_data(data, batch_size):
    while True: # this flag yields an infinite generator
        a, t, f, y, m = zip(*[next(data) for i in range(batch_size)])
        a = np.vstack(a)
        t = np.vstack(t)
        f = np.vstack(f)
        y = np.vstack(y)
        m = np.vstack(m)
        yield list(map(np.float32, (a, t, f, y, m)))

 
def compute_masked_accuracy(y_true, y_pred, mask):
    correct_preds = np.equal(np.argmax(y_true, 1), np.argmax(y_pred, 1))
    num_examples = float(np.sum(mask))
    correct_preds *= mask
    return np.sum(correct_preds) / num_examples
