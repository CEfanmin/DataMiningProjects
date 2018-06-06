import sys, os
os.environ['KERAS_BACKEND']='tensorflow'

if len(sys.argv) < 3:
    print('\nUSAGE: python %s <dataset_str> <gpu_id>' % sys.argv[0])
    sys.exit()
dataset = sys.argv[1]
gpu_id = sys.argv[2]

import numpy as np
from keras import backend as K
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from keras.models import load_model 
from utils_gcn import load_citation_data
from utils import generate_data, batch_data
from utils import compute_masked_accuracy
from models.ae import autoencoder_task
from layers.custom import DenseTied

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


print('\nLoading dataset {:s}...\n'.format(dataset))
try:
    adj, feats, y_train, y_val, y_test, mask_train, mask_val, mask_test = load_citation_data(dataset)
except IOError:
    sys.exit('Supported strings: {cora, citeseer, pubmed}')
feats = MaxAbsScaler().fit_transform(feats).tolil()
train = adj.copy()
adj.setdiag(1.0)
if dataset != 'pubmed':
    train.setdiag(1.0)


print('\nCompiling autoencoder model...\n')
encoder, ae = autoencoder_task(dataset, adj, feats, y_train)
adj = sp.hstack([adj, feats]).tolil()
train = sp.hstack([train, feats]).tolil()
print (ae.summary())

# Specify some hyperparameters
epochs = 10
train_batch_size = 256
val_batch_size = 256

print('\nFitting autoencoder model...\n')
train_data = generate_data(adj, train, feats,
                           y_train, mask_train, shuffle=True)
batch_data = batch_data(train_data, train_batch_size)
num_iters_per_train_epoch = adj.shape[0] / train_batch_size
y_true = y_val
mask = mask_val
for e in range(epochs):
    print('\nEpoch {:d}/{:d}'.format(e+1, epochs))
    curr_iter = 0
    train_loss = []
    for batch_a, batch_t, batch_f, batch_y, batch_m in batch_data:
        # Each iteration/loop is a batch of train_batch_size samples
        # batch_y = np.concatenate([batch_y, batch_m], axis=1)
        # res = ae.train_on_batch([batch_a, batch_f], [batch_t, batch_y])
        res = ae.train_on_batch([batch_a, batch_f], [batch_y])
        train_loss.append(res)
        curr_iter += 1
        if curr_iter >= num_iters_per_train_epoch:
            break
    train_loss = np.asarray(train_loss)
    train_loss = np.mean(train_loss, axis=0)
    print('Avg. training loss: {:s}'.format(str(train_loss)))

# decoded = ae.predict([adj.toarray(),feats.toarray()])
# print("decoded_nc: ", decoded.shape)
# np.savetxt("./data/decoded.csv", np.array(decoded), delimiter=",")

#     print('\nEvaluating validation set...')
#     decoded_nc = []
#     for step in range(int(adj.shape[0] / val_batch_size + 1)):
#         low = step * val_batch_size
#         high = low + val_batch_size
#         batch_adj = adj[low:high].toarray()
#         batch_feats = feats[low:high].toarray()
#         if batch_adj.shape[0] == 0:
#             break
        
#         decoded = ae.predict_on_batch([batch_adj, batch_feats])
#         print("decoded_nc shape is: ", decoded)
#         decoded_nc.append(decoded)

#     decoded_nc = np.vstack(decoded_nc)

    
#     node_val_acc = compute_masked_accuracy(y_true, decoded_nc, mask)
#     print('Node Val Acc {:f}'.format(node_val_acc))
# print('\nAll done.')

