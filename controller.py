import numpy as np

class Controller:
    def __init__(self, env):
        self.graph = env.graph
        self.env = env
    def reset(self, graph):
        self.graph = graph
    def act(self, obs):
        raise NotImplemented()

class

class NaiveController(Controller):
    def __init__(self, env):
        super().__init__(env)
        self.cache = {1: self.graph.get_rep()}
    def reset(self, graph):
        if graph != self.graph:
            self.cache = {1: graph.get_rep()}
        super().reset(graph)
    def gen_distance_matrices(self):
        matrix = self.graph.get_rep()
        yield matrix
        for i in range(self.graph.get_nb_nodes()):
            if i in self.cache:
                matrix = self.cache[i]
            else:
                matrix = matrix @ matrix
                self.cache[i] = matrix
            yield matrix
    def get_candidates(self, ref_pos):
        candidates = np.full((self.graph.get_nb_nodes(), self.graph.get_nb_nodes()), -1, dtype=float)
        candidates[ref_pos, ref_pos] = 0
        candidates = candidates[ref_pos]
        for i, m in enumerate(self.gen_distance_matrices()):
            m_for_cop_pos = m[ref_pos]
            candidates[np.logical_and(candidates == -1, m_for_cop_pos == 1)] = i + 1
            if np.all(candidates != -1):
                break
        return candidates

class NaiveRobber(NaiveController):
    def act(self, obs):
        candidates = self.get_candidates(self.env.cops_pos)
        if obs is None or -1 not in obs:
            # find the nodes the furthest away from all cops
            # then, find the best node among those (the column whose minimum is maximally away from the cops)
            best_candidate = np.argmax(np.min(candidates, axis=0))
            return [best_candidate]

        non_neighbours = self.graph.get_rep()[self.env.rob_pos[0]] == 0
        candidates = np.min(candidates, axis=0)
        candidates[non_neighbours] = -2*self.graph.get_nb_nodes()
        return [int(np.argmax(candidates))]

class NaiveCop(NaiveController):
    def act(self, obs):
        action = np.full((self.env.nb_cops,), -1)
        if obs is None or 1 not in obs:
            candidates = self.get_candidates(np.arange(self.graph.get_nb_nodes()))
            # find the shortest path to all nodes for all nodes
            # then, find the nodes with the best coverage (shortest distance to the most nodes)
            for cop in range(self.env.nb_cops):
                shortest_dist = np.min(candidates)
                best_row = np.argmax(np.count_nonzero(candidates == shortest_dist, axis=1))
                action[cop] = best_row
                row = candidates[best_row]
                candidates[row == 1] = 2*self.graph.get_nb_nodes()
                candidates[best_row] = 2*self.graph.get_nb_nodes()
            return action

        candidates = np.squeeze(self.get_candidates(self.env.rob_pos))
        non_neighbours = self.graph.get_rep()[self.env.cops_pos] == 0
        for cop in range(self.env.nb_cops):
            copy = candidates.copy()
            copy[non_neighbours[cop]] = 2*self.graph.get_nb_nodes()
            best_node = np.argmin(copy)
            action[cop] = best_node
            candidates[best_node] = 2*self.graph.get_nb_nodes()
            candidates[self.graph.get_rep()[best_node] == 1] = 2*self.graph.get_nb_nodes()
        return action

