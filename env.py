import math
import time

import gym
import numpy as np
import pyglet
from gym import spaces

from graph import Graph

REWARD_INVALID = -1000

WINDOW_W = WINDOW_H = 1000


# TODO allow cop to move faster than robber should it be needed (see paper). basically just give it d turns before the
# turn bool switches
class ShortestRouteEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, networkx_graph, origin, goal, weights=None, random_weights=(0, 10), make_finite=False):
        super(ShortestRouteEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.viewer = None
        self.is_finite = make_finite
        self.reset(networkx_graph, origin, goal, weights, random_weights, make_finite)

    def step(self, action):
        reward, done = self.graph.transition(action)
        if str(reward) == "False":  # action was invalid. Python is dumb and testing for "not reward" catches 0...
            reward = self.max_r * REWARD_INVALID
            print("INVALID!")

        return self.graph.position, reward, done, {}

    def reset(self, networkx_graph=None, origin=None, goal=None, weights=None, random_weights=(0, 10), make_finite=False):
        if networkx_graph is None:
            self.graph.reset(origin, goal)
        else:
            self.layout = None
            self.graph = Graph(networkx_graph, origin, goal, weights, random_weights, make_finite)
            self._make_PRAlpha()
            self.action_space = gym.spaces.Discrete(len(networkx_graph.nodes))
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Discrete(len(networkx_graph.nodes)),
                gym.spaces.Discrete(len(networkx_graph.nodes))
            ))
        return self.graph.position

    def __make_R(self, shortest):
        self.R = self.graph.adj_mat.copy().astype("float64")

        max_r = np.max(self.R)
        self.max_r = max_r
        # Make non-existent transitions very bad to use
        self.R[self.R == -1] = max_r * 10

        # make all weights non-zero                                                                       2
        # using a simple + 1 doesn't work because it breaks these arrangements (start at A and go to B): A--C
        #                                                                                             10 |  | 2
        #                                                                                                B--D
        #                                                                                                 5
        self.R = self.R + 1 / (10*len(self.graph.ngraph.nodes))

        self.R = -1 * self.R    # make rewards negative

        # The incentive to go to the goal is that once there, the agent can "spin" on it, gaining no negative rewards
        self.R[self.graph.goal, self.graph.goal] = 0

        if not shortest:
            self.R = 1/self.R

    # a bit weird, but allows us to redefine it easily in LongestRouteEnv
    def _make_R(self):
        self.__make_R(True)

    def _make_PRAlpha(self):
        adj_mat = self.graph.adj_mat.copy()

        # Completely deterministic. So, really, we duplicate the adj mat len(nodes) times
        adj_mat[adj_mat >= 0] = 1
        adj_mat[adj_mat < 0] = 0

        def range_except_i(n, i):
            t = []
            for j in range(n):
                if j != i:
                    t.append(j)
            return t

        P = []
        for n in range(len(self.graph.ngraph.nodes)):
            prob_mat_for_n = adj_mat.copy()
            # p of switching to non-connected node is 0
            prob_mat_for_n[:, range_except_i(len(self.graph.ngraph.nodes), n)] = 0

            # if action's node is not connected, stay in current node
            for i in range(len(self.graph.ngraph.nodes)):
                if 1 in prob_mat_for_n[i]:
                    continue
                else:
                    prob_mat_for_n[i, i] = 1

            P.append(prob_mat_for_n)

        self.P = np.array(P)
        self._make_R()
        self.alpha = np.array(range(len(self.graph.ngraph.nodes)))
        self.alpha[self.graph.origin] = 1  # initial distribution is just the origin

    def get_dijkstra(self):
        """
        Returns the optimal path and reward for the current problem.

        @return: dijkstra path, dijkstra reward
        """
        return self.graph.dijkstra_path, self.graph.dijkstra_rew

    def render(self, mode='human', human_reading_delay=1, layout_function="spring"):
        """
        Renders the state of the env for a human

        @param mode: only "human" is implemented
        @param human_reading_delay: delay for which the render stays visible (in seconds)
        @param layout_function: Any networkx layout works here https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
        """

        if self.graph.was_directed:
            layout_function = "shell"

        from gym.envs.classic_control import rendering
        import matplotlib.pyplot as plt
        import networkx as nx

        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
            self.viewer.width = WINDOW_W
            self.viewer.height = WINDOW_H
            self.viewer.window = pyglet.window.Window(width=self.viewer.width, height=self.viewer.height,
                                                      display=self.viewer.display, vsync=False, resizable=True)
        if self.layout is None:
            fun = getattr(nx, f"{layout_function}_layout")
            if layout_function == "spring":
                self.layout = fun(self.graph.ngraph, k=10/math.sqrt(self.graph.ngraph.order()))
            else:
                self.layout = fun(self.graph.ngraph)

        fig, ax = plt.subplots(figsize=(15, 15))
        fig.tight_layout()
        ax.axis("off")

        colors = np.array(["cyan"] * len(self.graph.ngraph.nodes))
        colors[list(self.graph.ngraph.nodes).index(self.graph.goal)] = "blue"
        colors[list(self.graph.ngraph.nodes).index(self.graph.position)] = "red"

        edge_colors = []
        for edge in self.graph.ngraph.edges:
            if edge in self.graph.path_bigram:
                edge_colors.append("pink")
            elif edge in self.graph.dijkstra_bigram:
                edge_colors.append("blue")
            else:
                edge_colors.append("black")


        nx.draw_networkx_nodes(self.graph.ngraph, self.layout, cmap=plt.get_cmap('jet'), node_color=colors, node_size=400)
        nx.draw_networkx_labels(self.graph.ngraph, self.layout)
        nx.draw_networkx_edges(self.graph.ngraph, self.layout, edge_color=edge_colors,
                               edgelist=[edge for edge in self.graph.ngraph.edges()], arrows=True,
                               connectionstyle='arc3, rad = 0.01')
        weights = nx.get_edge_attributes(self.graph.ngraph, "weight")

        weight_dict = {}
        for i in range(self.graph.adj_mat.shape[0]):
            for j in range(self.graph.adj_mat.shape[1]):
                w = self.graph.adj_mat[i,j]
                if w != -1:
                    weight_dict[(i,j)] = w

        nx.draw_networkx_edge_labels(self.graph.ngraph, self.layout,
                                     edge_labels=weight_dict, label_pos=0.25,
                                     font_size=12)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        self.viewer.imshow(image_from_plot)
        time.sleep(human_reading_delay)

    def close(self):
        self.viewer.window.close()

class LongestRouteEnv(ShortestRouteEnv):
    def _make_R(self):
        self.__make_R(1)
