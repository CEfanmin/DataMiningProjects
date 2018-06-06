import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda, add
from keras.models import Model
from keras import optimizers
from keras import backend as K
from layers.custom import DenseTied


def masked_categorical_crossentropy(y_true, y_pred):
    """ Categorical/softmax cross-entropy loss with masking """
    mask = y_true[:, -1]
    # y_true = y_true[:, :-1]
    loss = K.categorical_crossentropy(target=y_true,
                                      output=y_pred,
                                      from_logits=True)
    mask = K.cast(mask, dtype=np.float32)
    loss *= mask
    return K.mean(loss, axis=-1)

 
def autoencoder_task(dataset, adj, feats, labels, weights=None):
    adj = sp.hstack([adj, feats])
    h, w = adj.shape

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=np.float32, name='data')
 
    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(data)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    if dataset == 'pubmed':
        encoded = Dropout(rate=0.5, name='drop')(encoded)
    else:
        encoded = Dropout(rate=0.8, name='drop')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)
    
    ### Node classification ###
    feat_data = Input(shape=(feats.shape[1],))
    pred1 = Dense(labels.shape[1], activation='linear')(feat_data)
    pred2 = Dense(labels.shape[1], activation='linear')(decoded)
    prediction = add([pred1, pred2], name='prediction')

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)
    
    # compile the autoencoder
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    autoencoder = Model(inputs=[data, feat_data],
                        outputs=[prediction])
    
    # autoencoder.save("./models/aeModel.h5")
    autoencoder.compile(
            optimizer=adam,
            loss={'prediction': masked_categorical_crossentropy}
    )

    if weights is not None:
        autoencoder.load_weights(weights)

    return encoder, autoencoder

