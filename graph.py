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

def get_correct_adj_mat(ngaph):
    initial_adjacency_matrix = np.squeeze(
        np.asarray(
            nx.adjacency_matrix(ngaph, nodelist=sorted(ngaph.nodes()), weight=None).todense()
        )
    )
    adjacency_matrix = np.squeeze(
        np.asarray(
            nx.adjacency_matrix(ngaph, nodelist=sorted(ngaph.nodes())).todense()
        )
    )
    adjacency_matrix[initial_adjacency_matrix == 0] = -1  # replace non-edges by -1 instead of 0
    return adjacency_matrix

class Graph:
    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0,10), make_horizon=False):
        networkx_graph = networkx_graph.copy()
        self.was_directed = nx.is_directed(networkx_graph)
        networkx_graph = networkx_graph.to_directed()

        networkx_graph = nx.convert_node_labels_to_integers(networkx_graph, label_attribute="old_name")

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

        networkx_graph = networkx_graph if not make_horizon else self._make_horizons(networkx_graph, origin, goal)

        self.make_horizon = make_horizon
        self.ngraph = networkx_graph
        self.adj_mat = get_correct_adj_mat(networkx_graph)
        self.adjacent_indices = [np.nonzero(self.adj_mat[i] != -1)[0] for i in range(self.adj_mat.shape[0])]
        self._set_problem(origin, goal)


    def _make_horizons(self, ngraph, origin, goal):
        # TODO test if ngraph is already horizon-compatible

        all_paths = sorted(nx.all_simple_paths(ngraph, origin, goal), key=len, reverse=True)
        max_len = len(all_paths[0])
        adj_mat = get_correct_adj_mat(ngraph)

        _new_name = [max_len+1]
        def get_new_name(_new_name=_new_name):
            temp = _new_name[0]
            while temp == goal or temp == origin:
                temp += 1
            _new_name[0] = temp + 1
            return temp

        # go-go gadgeto extendo paths
        for path in all_paths:
            while len(path) < max_len:
                path.insert(-1, get_new_name())

        # rename paths
        _new_name[0] = 0 # reset renamer
        flow_graph = nx.DiGraph()
        flow_graph.add_node(origin)
        flow_graph.add_node(goal)
        for path in all_paths:
            new_u_name = origin
            for i, uv in enumerate(make_bigram(path)):
                u, v = uv
                new_v_name = v
                if v != goal:
                    new_v_name = get_new_name()
                    flow_graph.add_node(new_v_name)

                w = 0
                if u < ngraph.order():
                    if v < ngraph.order():
                        w = adj_mat[u,v]
                    else:
                        w = adj_mat[u, goal]
                flow_graph.add_edge(new_u_name, new_v_name, weight=w)
                new_u_name = new_v_name
        print("BEFORE COLLAPSE:", flow_graph.order())

        # collapse end
        front = goal
        while True:
            neighs = list(flow_graph.predecessors(front))
            for neigh in neighs.copy():
                if flow_graph[neigh][front]["weight"] != 0:
                    neighs.remove(neigh)

            if len(neighs) <= 1:
                break

            front = neighs[0]
            for neigh in neighs[1:]:
                for pred in flow_graph.predecessors(neigh):
                    flow_graph.add_edge(pred, front, weight=flow_graph[pred][neigh]["weight"])
                flow_graph.remove_node(neigh)

        # final_relabeling
        _new_name[0] = 0
        for n in list(flow_graph.nodes):
            if n in [origin, goal]:
                continue
            nx.relabel_nodes(flow_graph, {n: get_new_name()}, copy=False)
        print("AFTER COLLAPSE:", flow_graph.order())

        return flow_graph

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
