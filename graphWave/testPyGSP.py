import numpy as np
from matplotlib import pyplot as plt
from pygsp import graphs, filters, plotting, utils

# parameters
taus = [10,25,50]

def dataSource():
    '''
    source graph
    '''
    G = graphs.Bunny()
    # G.plot()
    # plt.show()
    return G

def heatDiffusion(G, taus=taus):
    '''
    heat diffusion visualization
    '''
    g = filters.Heat(G,taus)
    s = np.zeros(G.N)
    DELTA = 20
    s[DELTA] = 1
    s = g.filter(s, method='chebyshev')

    fig = plt.figure(figsize = (10,3))
    for i in range(g.Nf):
        ax = fig.add_subplot(1, g.Nf, i+1, projection='3d')
        G.plot_signal(s[:,i], ax=ax)
        title = r'Heat diffusion, $\ tau={}$'.format(taus[i])
        _ = ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()

G = dataSource()
heatDiffusion(G, taus=taus)


