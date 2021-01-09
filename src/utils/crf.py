import itertools
import numpy as np
import torch


def get_graph_idx(edges, cliques):
    """
    Get all necessary indices for the higher-order CRF from lists of edges and cliques.

    Args:
        edges:           List of tuples, each tuple containing two nodes, e.g. [(0,1),(0,2),(1,2)],
                         All edges assumed to be undirected, so that (0,1)==(1,0)
        cliques:         Same as edges, but contains all cliques in the graph, e.g. [(0,1,2)]
    Returns:
        edge_pair_index: Holds every pair of edges that occurs in cliques
        j_edge_index:    Holds for every edge the indices of edge pairs that form a clique of size 3 together
        clique_index:    Holds for every clique the indices of the three edges in the clique
    """

    if isinstance(edges,torch.Tensor):
        edges = edges.tolist()
        edges = [tuple(l) for l in edges]

    if isinstance(cliques,torch.Tensor):
        cliques = cliques.tolist()
        cliques = [tuple(l) for l in cliques]

    # assign index to edges and cliques
    edge_index = dict(zip(edges,np.arange(0,len(edges),1)))
    clique_index = []
    for c in cliques:

        individual_edges = list(itertools.combinations(c,2))
        # may be in wrong order, reverse if a combination is not in the edge_index
        for n, ie in enumerate(individual_edges):
            if ie not in edge_index:
                individual_edges[n] = ie[::-1]  # reverse

        clique_ints = list(map(edge_index.get,individual_edges))
        clique_index.append(clique_ints)

    # create edge_pair_index which holds every edge pair that occurs in a clique
    edge_pair_index = torch.tensor(np.concatenate([np.array(list((itertools.combinations(c,2)))) for c in clique_index]),dtype=torch.long).t().contiguous()

    # prepare efficient passing of messages
    # for each edge get array of their clique's other edge combinations
    # e.g. if edge is (0,1) and clique is (0,1,2) j_edges will hold index for edges (0,2) and (1,2)
    # every edge can belong to variable number of cliques
    j_edges = get_j_edges(np.array(clique_index))
    # make tuples because python dict keys may not be lists
    j_edges = [list((tuple(j) for j in l)) for l in j_edges]
    js_list = edge_pair_index.t().tolist()
    pairs_dict = dict(zip(list(map(tuple,js_list)),np.arange(0,len(js_list),1)))

    j_edge_index = convert(j_edges,pairs_dict)

    return edge_pair_index, j_edge_index, clique_index


def get_j_edges(x):

    x_flat = x.ravel()
    ix_flat = np.argsort(x_flat)
    u, ix_u = np.unique(x_flat[ix_flat], return_index=True)
    ix_ndim = np.unravel_index(ix_flat, x.shape)[0]

    # for each edge get the position of cliques it is part of
    pos = np.split(ix_ndim, ix_u[1:])

    try:

        # check if pos can be transformed to 2-d numpy array (only the case if every edge occurs in same amount of cliques)
        np.array(pos).shape[1]

        # if every edge occurs in same amount of cliques use vectorized approach
        x_2d = x[pos].ravel().reshape(-1,3)

        # for each clique get position of current edge
        pos_in_c = np.unravel_index(ix_flat, x.shape)[1]

        m,n = x_2d.shape
        j_edges = x_2d[np.arange(n) != pos_in_c[:,None]].reshape(m,-1)
        j_edges = np.split(j_edges, ix_u[1:])

    except:
        # remove the current edge to get j_edges (has to be list as each edge can belong to variable number of cliques)
        j_edges = [x[pos[e]][x[pos[e]]!=e].reshape(-1,2) for e in u]

    return j_edges


def convert(l, d):
    return [convert(x, d) if isinstance(x, list) else d.get(x, x) for x in l]