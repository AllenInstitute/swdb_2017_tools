# Load required modules
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os
import seaborn as sns

def create_dendrogram(d,nodes = None,leaves = None, inner_nodes = None):
    '''
    Parameters
    ----------
    d : dictionary
        a dictionary that specifies the tree structure
        ex: d = { 0: [1, 2], 1: [3, 4], 2:[5,6],
                 3:['bird','bug','mammal','reptile'], 4:['flower','leaf','tree'],
                 5:['building','other'], 6:['ground','landscape'],
                 'mammal':[],'bird':[],'reptile':[],'bug':[],'flower':[],'tree':[],
                 'leaf':[],'building':[],'other':[],'landscape':[],'ground':[]}
    plot : boolean
        should a plot be generated
    nodes : list
        a list with the names of all the nodes in the tree
    leaves : set
        a set with the names of all the leaves in the tree
    inner_nodes : list
        a list with the names of all the inner nodes in the tree
        
    Returns
    -------        
    Z: list
        linkage
    
    Dendogram plot
    
    Colleen Schneider 8/25/17
    '''
    if nodes == None:
        # Construct the graph/hierarchy
        G           = nx.DiGraph(d)
        nodes       = G.nodes()
        leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
        inner_nodes = sorted([ n for n in nodes if G.out_degree(n) > 0 ])
    
    # Compute the size of each subtree
    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(d[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += d[v]
    
        subtree[u] = sorted(children & leaves)
    
    inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last
    
    # Construct the linkage matrix
    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = d[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            Z.append([i, j, float(len(subtree[n])), len(z)]) # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = sorted(list(z))
            x = z
            k += 1
    
       
    # Visualize
    sns.set_context("talk")
    sns.set_style("white")
    R = dendrogram(Z, labels=leaves,leaf_font_size = 30,orientation = 'left')
    plt.figure(figsize = [15,5])
    
    return Z

