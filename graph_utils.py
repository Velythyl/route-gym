import copy

import networkx as nx
import numpy as np

class Graph:
    def __init__(self, networkx_graph):
        self.igraph = networkx_graph
        self._rep = np.squeeze(
            np.asarray(
                nx.adjacency_matrix(self.igraph, nodelist=sorted(self.igraph.nodes())).todense()
            )
        )
        self._attr = np.zeros(self._rep.shape[0], dtype=int)

    def set_cr(self, nodes, is_cop):
        self._attr[self._attr == (1 if is_cop else -1)] = 0
        self._attr[nodes] += 1 if is_cop else -1

    def get_attr(self):
        return self._attr

    def get_rep(self):
        return self._rep

    def get_nb_nodes(self):
        return len(self._attr)