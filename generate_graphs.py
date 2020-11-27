import networkx as nx
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import count

import gc
import builder

def save_graph(graph, layout,file_name):
    groups = set(nx.get_node_attributes(graph,'country').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = graph.nodes()
    colors = [mapping[graph.node[n]['country']] for n in nodes]
    
    from matplotlib import pylab
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    layouts = {'spring':nx.spring_layout, 
               'kamada_kawai':nx.kamada_kawai_layout, 
               'shell':nx.shell_layout, 
               'spectral':nx.spectral_layout, 
               'circular':nx.circular_layout}
    fig = plt.figure(1)
    pos = layouts[layout](graph)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, 
                                with_labels=False, node_size=15, cmap=plt.cm.jet)
    plt.title('Russian (Red) & Iranian (Blue) Core Troll Accounts')
    plt.axis('off')
    plt.savefig(file_name,bbox_inches="tight")
    plt.close()
    del fig, groups, mapping, nodes, colors
    del ec, nc

users, interactions = builder.full_network()
core = [u for u in users if u[1]['account'] == 'removed']
users = [u[0] for u in core]
interactions = interactions[interactions['source'].isin(users) & interactions['target'].isin(users)]

directed_core = nx.DiGraph()
directed_core.add_nodes_from(core)
directed_core.add_weighted_edges_from(interactions[['source','target','total']].values, weight='total')

for l in ['spring','kamada_kawai','shell','spectral','circular']:
    save_graph(directed_core, l, l+'.pdf')
    gc.collect()
