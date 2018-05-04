# -*- coding: utf-8 -*-
"""

This file contains the script for defining characteristic functions and using them
as a way to embed distributional information in Euclidean space
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def characteristic_function(sig,t,plot=False, taus=1, node=1):
    ''' function for computing the characteristic function associated to a signal at
        a point/ set of points t:
            f(sig,t)=1/len(sig)* [sum_{s in sig} exp(i*t*s)]
    INPUT:
    ===========================================================================
    sig   :      signal over the graph (vector of coefficients)
    t     :      values at which the characteristic function should be evaluated
    plot  :      boolean: should the resulting point/set of points be plotted
    
    OUTPUT:
    ===========================================================================
    f     :      empirical characteristic function
    '''
    f=np.zeros((len(t),3))
    if type(t) is list:
        f=np.zeros((len(t),3))
        f[0,:]=[0,1,0]
        vec1=[np.exp(complex(0,sig[i])) for i in range(len(sig))]
        for tt in range(1,len(t)):
            f[tt,0]=t[tt]
            vec=[x**t[tt] for x in vec1]
            c=np.mean(vec)
            f[tt,1]=c.real
            f[tt,2]=c.imag
        if plot==True:
            plt.figure()
            plt.plot(f[:,1],f[:,2], c='g')
            plt.title("characteristic function of the distribution")
            plt.xlabel('real part')
            plt.ylabel('image part')
            plt.savefig('../figure/ChaFun_taus%s_node%s.png'%(taus, node))

    else:
        c=np.mean([np.exp(complex(0,t*sig[i])) for i in range(len(sig))])
        f=[t,np.real(c),np.imag(c)]
    return f

def featurize_characteristic_function(heat_print,t=[],nodes=[]):
    ''' same function as above, except the coefficient is computed across all scales and concatenated in the feature vector
    Parameters
    ----------
    heat_print
    t:             (optional) values where the curve is evaluated
    nodes:         (optional at  which nodes should the featurizations be computed (defaults to all)
    
    Returns
    -------
    chi:            feature matrix (pd DataFrame)
    '''
    
    if len(t)==0:
        t=range(0,100,5)
        t+=range(85,100)
        t.sort()
        t=np.unique(t)
        t=t.tolist()
    if len(nodes)==0:
        nodes=range(heat_print[0].shape[0])
    
    chi=np.empty((len(nodes),2*len(t)*len(heat_print)))
    for tau in range(len(heat_print)):
        sig=heat_print[tau]
        for i in range(len(nodes)):
            ind=nodes[i]
            s=sig.iloc[:,ind].tolist()
            c=characteristic_function(s,t,plot=True, taus=tau, node=i)
            # Concatenate all the features into one big vector
            chi[i,tau*2*len(t):(tau+1)*2*len(t)]= np.reshape(c[:,1:],[1,2*len(t)])
    # chi=pd.DataFrame(chi, index=[nodes[i] for i in range(len(nodes))])
    return chi
    
