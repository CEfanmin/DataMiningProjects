import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns

def load_data():
    '''load data'''
    raw_data = pd.read_csv('../emb/test3-negative_sampling.csv')
    feture_data = raw_data.iloc[:,0:2]
    id_data = raw_data['id'].tolist()
    return feture_data, id_data

def visulization(feture_data, id_data):
    '''Visualize'''
    # map_tsne = TSNE(n_components=2, random_state=14)
    # visualized_features = pd.DataFrame(map_tsne.fit_transform(feture_data),columns=['x1','x2'])
    # print(visualized_features)

    fig, ax = plt.subplots()
    plt.scatter(feture_data['x1'], feture_data['x2'],marker='o',s=150, c=id_data)
    plt.title('graph wave embedding features')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')

    for i,txt in enumerate(np.arange(1,69)):
        ax.annotate(txt,(feture_data['x1'][i],feture_data['x2'][i]),ha='right')
    
    # fig.savefig('../karate.png')
    ax.grid(True)
    plt.show()

feture_data, id_data = load_data()
visulization(feture_data, id_data)
