import copy

import networkx as nx
import numpy as np

def make_bigram(list):
    return [(list[i], list[i + 1]) for i in range(len(list) - 1)]

# translate to numbers if need be
def remap_to_numbers(networkx_graph):
    mapping = {}
    for i, node in enumerate(networkx_graph.nodes):
        try:
            if int(node) == node:
                break
        except ValueError:
            pass

        mapping[node] = i
    return nx.relabel_nodes(networkx_graph, mapping)

class Graph:
    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0,10)):
        networkx_graph = networkx_graph.copy()
        self.was_directed = nx.is_directed(networkx_graph)
        networkx_graph = networkx_graph.to_directed()

        networkx_graph = remap_to_numbers(networkx_graph)

        # initial, non-weighted adjacency matrix because when a weight is zero we confuse it for there not being an edge
        initial_adjacency_matrix = np.squeeze(
            np.asarray(
                nx.adjacency_matrix(networkx_graph, nodelist=sorted(networkx_graph.nodes()), weight=None).todense()
            )
        )

        if nx.is_weighted(networkx_graph) and weights is not None:
            print("WARNING: your weights will be ignored.")

        if not nx.is_weighted(networkx_graph):  # First, make sure the graph is weighted
            edges = networkx_graph.edges()

            dico = {}

            if weights is not None:
                for e1, e2 in edges:
                    try:    # Runs once, doesn't have to be optimized
                        dico[(e1, e2)] = weights[e1][e2]
                    except:
                        try:
                            dico[(e1, e2)] = weights[(e1,e2)]
                        except:
                            raise Exception("The weights passed to Graph must either be indexed by (edge one, edge two) tuples or by edge one, and then edge two."
                                            "\n\nSo either weights[(e1,e2)]=value or weights[e1][e2]=value.")

            for e1e2 in edges:
                if e1e2 not in dico:
                    dico[e1e2] = np.random.randint(random_weights[0], random_weights[1], size=1)[0]
            nx.set_edge_attributes(networkx_graph, dico, "weight")

        adjacency_matrix = np.squeeze(
            np.asarray(
                nx.adjacency_matrix(networkx_graph, nodelist=sorted(networkx_graph.nodes())).todense()
            )
        )
        adjacency_matrix[initial_adjacency_matrix == 0] = -1    # replace non-edges by -1 instead of 0

        self.ngraph = networkx_graph
        self.adj_mat = adjacency_matrix
        self.adjacent_indices = [np.nonzero(self.adj_mat[i] != -1)[0] for i in range(self.adj_mat.shape[0])]
        self._set_problem(origin, goal)

    def _set_problem(self, origin, goal):
        self._set_position(origin)
        self.origin = origin
        self.goal = goal
        self.path = [origin]
        self.path_bigram = []
        self.dijkstra_path = nx.dijkstra_path(self.ngraph, origin, goal)
        self.dijkstra_bigram = make_bigram(self.dijkstra_path)
        self.dijkstra_rew = sum([self.adj_mat[(e1, e2)] for e1, e2 in self.dijkstra_bigram])

    def reset(self, origin=None, goal=None):
        if origin is None:
            origin = self.origin
        if goal is None:
            goal = self.goal
        self._set_problem(origin, goal)

    def _set_position(self, pos):
        self.position = pos

    def transition(self, new_pos):
        self.path.append(new_pos)
        self.path_bigram = make_bigram(self.path)

        if new_pos not in self.adjacent_indices[self.position]:
            print(f"{new_pos} not in {self.adjacent_indices[self.position]}")
            import time
            time.sleep(100)
            return False, False

        reward = self.adj_mat[self.position, new_pos]
        self._set_position(new_pos)

        done = self.position == self.goal

        return reward, done