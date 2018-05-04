import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from graphwave import graphwave

def read_graph():
    '''read graph'''
    G = nx.read_edgelist('../data/karate-mirrored.edgelist',nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    G = G.to_undirected()
    return G

def load_data():
    '''load data'''
    raw_data = pd.read_csv('../data/karate-emb.csv')
    feture_data = raw_data.iloc[:,0:2]
    id_data = raw_data['id'].tolist()
    return feture_data, id_data

def visulization(feture_data, id_data):
    '''Visualize'''
    fig, ax = plt.subplots()
    plt.scatter(feture_data['x1'], feture_data['x2'],marker='o',s=150, c=id_data)
    plt.title('graph wave embedding features')
    plt.xlabel('1st feature')
    plt.ylabel('2nd feature')

    for i,txt in enumerate(np.arange(1,69)):
        ax.annotate(txt,(feture_data['x1'][i],feture_data['x2'][i]),ha='left')
    ax.grid(True)
    plt.show()

if __name__=="__main__":
    '''pipeline'''
    nx_G = read_graph()
    nx.draw(nx_G, pos=nx.spring_layout(nx_G))
    # plt.show()
    chi, heat_print, taus=graphwave(nx_G, 'automatic', verbose=False)
    print('taus is:',taus)
    print(np.array(chi).shape)
    # pca=PCA(n_components=2)
    # trans_data=pca.fit_transform(StandardScaler().fit_transform(chi))
    # print(trans_data.shape)
    # pd.DataFrame(trans_data).to_csv('../data/karate-emb.csv')

    feture_data, id_data = load_data()
    visulization(feture_data, id_data)
