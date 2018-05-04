# -*- coding: utf-8 -*-
"""
All functions for computing the wavelet distributions 
"""
import pygsp
import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt


def heat_diffusion(G,taus=[1, 10, 25, 50],diff_type="immediate",b=1,type_graph="pygsp"):
    '''
        This method computes the heat diffusion waves for each of the nodes
     INPUT:
    -----------------------
    G: Graph, can be of type networkx or pygsp
    taus: list of 4 scales for the wavelets. The higher the tau, the better the spread
    type: tyoe of the  graph (networkx or pygsp)

    OUTPUT:
    -----------------------
    list_of_heat_signatures: list of 4 pandas df corresponding to the heat signature
    for each diffusion wavelet
    '''

    if type_graph=="pygsp":
        A=G.W
        N=G.N
        if diff_type=="delayed":
            Hk = pygsp.filters.DelayedHeat(G, b,taus, normalize=False)
        elif diff_type=="mexican":
            Nf=6
            Hk = pygsp.filters.MexicanHat(G, Nf)
        elif diff_type=="wave":
            Hk = pygsp.filters.Wave(G, taus, normalize=False)
        else:
            Hk = pygsp.filters.Heat(G, taus, normalize=False)

    elif type_graph=="nx":
        A=nx.adjacency_matrix(G)
        N=G.number_of_nodes()
        Gg = pygsp.graphs.Graph(A,lap_type='normalized')
        if diff_type=="delayed":
            Hk = pygsp.filters.DelayedHeat(Gg,b, taus, normalize=False)
        elif diff_type=="mexican":
            Nf=6
            Hk = pygsp.filters.MexicanHat(Gg, Nf)
        elif diff_type=="wave":
            Hk = pygsp.filters.Wave(Gg, taus, normalize=True)
        else:
            Hk = pygsp.filters.Heat(Gg, taus, normalize=False)
            # s = Hk.localize(N // 2)
            # fig, axes = plt.subplots(1, 1)
            # Hk.plot(ax=axes)

    else:
        print "graph type not recognized"
        return False
    
    heat={i:pd.DataFrame(np.zeros((N,N)), index=range(N)) for i in range(len(taus))}   
    for v in range(N):
            ### for each node v , create a signal that corresponds to a Dirac of energy
            ### centered around v and which propagates through the network
            f=np.zeros(N)
            f[v]=1
            Sf_vec = Hk.analyze(f) ### creates the associated heat wavelets
            Sf = Sf_vec.reshape((Sf_vec.size/len(taus), len(taus)), order='F')
            for  i in range(len(taus)):
                heat[i].iloc[:,v]=Sf[:,i] ### stores in different dataframes the results
    return [heat[i] for i in range(len(taus))]
#return pd.DataFrame.from_dict(heat)
