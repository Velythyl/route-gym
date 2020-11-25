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
    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0,10), make_finite=False):
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

        networkx_graph = self._define_finite_problem(networkx_graph, origin,goal) if make_finite else networkx_graph

        self.ngraph = networkx_graph
        self.adj_mat = get_correct_adj_mat(networkx_graph)
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

    def _define_finite_problem(self, ngraph, origin, goal):
        tree = nx.bfs_tree(ngraph, origin)
        orig_tree = tree.copy()

        adj_mat = get_correct_adj_mat(ngraph)

        # bfs doesn't conserve weight...
        for u, v in tree.edges:
            tree[u][v]["weight"] = adj_mat[u,v]
        ngraph = ngraph.copy()
        ngraph_saved = ngraph.copy()

        def get_layer(n):
            front = [origin]
            for i in range(n):
                new_front = []
                for f in front:
                    new_front += list(nx.neighbors(orig_tree, f))
                front = new_front
            return front

        def problematic_neighbours(i):
            # returns all the bad neighbours at distance i of origin
            previous_layer = get_layer(i-1)
            this_layer = get_layer(i)

            all_neighbours = set(nx.utils.flatten([list(nx.neighbors(ngraph, node)) for node in this_layer]))
            problematic_neighs = (all_neighbours - set(previous_layer)).intersection(this_layer)
            return problematic_neighs

        _new_name = [len(tree.nodes)]
        def get_new_name(_new_name=_new_name):
            temp = _new_name[0]
            _new_name[0] = _new_name[0] + 1
            return temp

        def draw(): # debug
            import matplotlib.pyplot as plt

            pos = nx.kamada_kawai_layout(tree)
            nx.draw_networkx_nodes(tree, pos, cmap=plt.get_cmap('jet'))
            nx.draw_networkx_labels(tree, pos)
            nx.draw_networkx_edges(tree, pos, edgelist=tree.edges(), arrows=True)
            plt.show()

        max_path_len = max([len(path) for path in nx.all_simple_paths(ngraph, origin, goal)])

        mapping = {}
        for distance in range(1, max_path_len):
            #draw()
            bad_neighs = problematic_neighbours(distance)

            if len(bad_neighs) == 0:
                continue

            # Step 1: rename
            for bad_neigh in bad_neighs:
                mapping[bad_neigh] = get_new_name()
            tree = nx.relabel_nodes(tree, mapping)

            # Step 2: extension
            tree.add_nodes_from(bad_neighs)
            for old_name, new_name in mapping.items():
                for good_neighbour in list(nx.neighbors(tree, new_name)):   # ??
                    tree.remove_edge(new_name, good_neighbour)
                    tree.add_edge(old_name, good_neighbour)
                tree.add_edge(new_name, old_name, weight=0)

            # Step 3: add bad edges
            for old_name in bad_neighs:
                nodes_to_join = set(nx.neighbors(ngraph, old_name)).intersection(bad_neighs)
                for joinee in nodes_to_join:
                    weight = adj_mat[old_name, joinee]
                    tree.add_edge(mapping[old_name], joinee, weight=weight)
                    ngraph.remove_edge(old_name, joinee)

        # Step 4: update old edges to new nodes
        """
        edges = set(tree.edges)
        attrs = nx.get_edge_attributes(tree, "weight")
        edges = edges.difference(set(attrs.keys()))
        inv_mapping = {v: k for k, v in mapping.items()}
        for u, v in edges:
            new_u = mapping.get(u, u)
            new_v = mapping.get(v, v)
            old_u = inv_mapping.get(u, u)
            old_v = inv_mapping.get(v, v)

            tree[u][v]["weight"] = get_correct_adj_mat(ngraph)[old_u,old_v]
        """



        return tree

        # NO
        """

        flow_graph = nx.DiGraph()
        visited = set()
        last_front = []
        front = set([self.origin])

        new_node_name = len(self.ngraph)
        renames = {}

        while len(visited) < len(self.ngraph):
            new_front = set()
            while len(front) > 0:
                visitee = list(front)[0]
                front.remove(visitee)

                flow_graph.add_node(visitee)
                def add_edge(u1, u2, w=None):
                    if w is None:
                        try:
                            w = self.adj_mat[u1, u2]
                        except:
                            w = 0
                    flow_graph.add_edge(u1, u2, weight=w)

                problematic_neighbours = list(nx.neighbors(self.ngraph, visitee))
                for neigh in problematic_neighbours.copy():
                    if neigh in visited:
                        add_edge(visitee, neigh)
                        problematic_neighbours.remove(neigh)
                    elif neigh not in front:
                        new_front.add(neigh)
                        problematic_neighbours.remove(neigh)

                for bad_neigh in problematic_neighbours:    # these are both neigh of visitee and in the front
                    # the bad neigh is added under a different name, and the true name is only used for the disentangled
                    # node
                    renames[bad_neigh] = new_node_name
                    flow_graph.add_node(new_node_name)
                    flow_graph.add_node(bad_neigh)
                    flow_graph.add_edge(visitee, new_node_name, w=self.adj_mat[visitee, bad_neigh])
                    flow_graph.add_edge(new_node_name, bad_neigh)

                    for bad_neigh_neigh in problematic_neighbours:
                        if bad_neigh in nx.neighbors(self.ngraph, bad_neigh):


                visited.add(visitee)
            front = new_front

        nx.neighbors(self.ngraph, self.origin)

        tree = nx.bfs_tree(self.ngraph, self.origin)
        temp=0
        """



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