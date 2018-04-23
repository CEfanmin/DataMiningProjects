import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns

def load_data():
    '''load data'''
    raw_data = pd.read_csv('./emb/test.csv')
    feture_data = raw_data.iloc[:,0:128]
    id_data = raw_data['id'].tolist()
    return feture_data, id_data

def visulization(feture_data, id_data):
    '''Visualize'''
    map_tsne = TSNE(n_components=2, random_state=14)
    visualized_features = pd.DataFrame(map_tsne.fit_transform(feture_data),columns=['x1','x2'])
    print(visualized_features)

    fig, ax = plt.subplots()
    plt.scatter(visualized_features['x1'], visualized_features['x2'],marker='o',c=id_data, label=id_data)
    plt.title('the features')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')
    ax.grid(True)
    plt.show()

feture_data, id_data = load_data()
visulization(feture_data, id_data)
