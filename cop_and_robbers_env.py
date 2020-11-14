import time

import gym
import numpy as np
import pyglet
from gym import spaces

REWARD_END_WL = 1000
REWARD_STEP_WL = 1
REWARD_INVALID = -10

WINDOW_W = WINDOW_H = 600


# TODO allow cop to move faster than robber should it be needed (see paper). basically just give it d turns before the
# turn bool switches
class CopRobEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, graph, nb_cops, auto_robber_class=None, auto_cop_class=None):
        super(CopRobEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.viewer = None
        self.layout = None
        self.cops = None  # ugly, fixme
        self.robber = None  # ugly, fixme
        self.nb_cops = nb_cops
        self.reset(graph)

        self.cops = None if auto_cop_class is None else auto_cop_class(self)
        self.robber = None if auto_robber_class is None else auto_robber_class(self)

    def step(self, action):  # action is nb-cops-sized or 1-sized
        """
        The returned obs only contains the new attributes of the nodes. The trainer should keep the repr of the graph
        on its own (this is for performance reasons).

        :param action:
        :return: obs, rew, done, info
        """
        reward = 0
        done = False

        action = np.array(action)

        def old_pos(set=None):
            if set is None:
                return self.cops_pos if self.is_cops_turn else self.rob_pos
            else:
                if self.is_cops_turn:
                    self.cops_pos = action
                else:
                    self.rob_pos = action

        invalids = []

        if self.is_first_turn:
            self.graph.set_cr(action, self.is_cops_turn)
        else:
            edges = self.graph.get_rep()[old_pos(), action]
            invalids = edges != 1
            invalids[action == old_pos()] = False
            invalids = np.where(invalids == True)[0]
            if invalids.shape[0] != 0:
                action[invalids] = old_pos()[invalids]  # correct action
            self.graph.set_cr(action, self.is_cops_turn)

        old_pos(action)
        if not self.is_cops_turn and self.is_first_turn:
            self.is_first_turn = False
        self.is_cops_turn = not self.is_cops_turn
        if self.rob_pos is not None and self.rob_pos[0] in self.cops_pos:
            print("Cops won")
            done = True
            reward += (1 if self.is_cops_turn else -1) * REWARD_END_WL

        reward += (-1 if self.is_cops_turn else +1) * REWARD_STEP_WL
        reward -= len(invalids) * REWARD_INVALID

        observation = self.graph.get_attr()

        if self.is_cops_turn:
            self.cops_rew += reward
        else:
            self.rob_rew += reward

        if not done:
            if self.is_cops_turn and self.cops is not None:
                observation, _, done, _ = self.step(self.cops.act(observation))
            elif not self.is_cops_turn and self.robber is not None:
                observation, _, done, _ = self.step(self.robber.act(observation))
        return observation, reward, done, {}

    def reset(self, graph=None):
        if graph is not None:
            self.graph = graph
            self.viewer = None
            self.layout = None

            if self.cops is not None:
                self.cops.reset(graph)
            if self.robber is not None:
                self.robber.reset(graph)

        # Initially, cops & robbers can choose position, so space is basically all the graph
        self.cop_action_space = spaces.Box(0, self.graph.get_nb_nodes(), shape=(self.nb_cops,), dtype=int)
        self.rob_action_space = spaces.Box(0, self.graph.get_nb_nodes(), shape=(1,), dtype=int)
        self.observation_space = spaces.Discrete(self.graph.get_nb_nodes())
        self.is_cops_turn = True
        self.is_first_turn = True
        self.rob_pos = None
        self.cops_pos = None
        self.rob_rew = 0
        self.cops_rew = 0

    def render(self, mode='human', human_reading_delay=1):
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
            self.layout = nx.spring_layout(self.graph.igraph)

        G = self.graph.igraph
        fig, ax = plt.subplots()

        colors = np.array(["white"] * len(G.nodes))
        if self.cops_pos is not None:
            colors[self.cops_pos] = "cyan"
        if self.rob_pos is not None:
            colors[self.rob_pos] = "red"

        nx.draw_networkx_nodes(G, self.layout, cmap=plt.get_cmap('jet'), node_color=colors, node_size=500)
        nx.draw_networkx_labels(G, self.layout)
        nx.draw_networkx_edges(G, self.layout, edgelist=[edge for edge in G.edges()], arrows=False)

        ax.set_title("Robber's reward:" + str(self.rob_rew) + "; Cops' Reward:" + str(self.cops_rew))

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        self.viewer.imshow(image_from_plot)
        time.sleep(1)