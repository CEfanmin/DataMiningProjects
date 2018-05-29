import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_data(filename, filename_label):
    '''
    load raw data
    '''
    raw_data = pd.read_csv(filename)
    feature_data = raw_data.iloc[:, 0:64]
    onehot_label = np.array(pd.read_csv(filename_label).iloc[:, 0:7])
    print('onehot_label shape is:', onehot_label.shape)
    integer_label = np.argmax(onehot_label, axis=1)
    print('label: ',integer_label)
    return feature_data, integer_label

def visulization(feature_data, label_class):
    '''
    visulization data
    '''
    map_tsne = TSNE(n_components=2, random_state=14)
    visualized_features = pd.DataFrame(map_tsne.fit_transform(feature_data),columns=['x1','x2'])
    
    ax = None
    for c in range(7):
        ax = visualized_features.iloc[
            list(np.where(label_class == c)[0]), :
        ].plot(
            kind='scatter',x='x1',y='x2', color=sns.color_palette('husl', 7)[c],label='class %d' % c, ax =ax
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('first layer embedded features')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    plt.show()


feature_data, label_class = load_data('./data/1rd_layer_output_2000.csv', './data/label.csv')
visulization(feature_data, label_class)