import time

import networkx as nx

from controller import NaiveRobber, NaiveCop
from cop_and_robbers_env import CopRobEnv
from graph_utils import Graph

networkx_graph = nx.frucht_graph()
graph = Graph(networkx_graph)
env = CopRobEnv(graph, 2, auto_robber_class=NaiveRobber)

robber = NaiveRobber(env)
cop = NaiveCop(env)

env.render()
done = False
obs = None
while not done:
    obs, _, done, _ = env.step(cop.act(obs))
    if done:
        break
    env.render()
    #obs, _, done, _ = env.step(robber.act(obs))
    #env.render()
exit()

env.render()
for act in [[0, 1], [3], [4,5], [4]]:
    env.step(act)
    env.render()
exit()


env.render()
env.step([0, 1])    # init cops
env.render()
env.step([3])   # init rob
env.render()
env.step([4,5]) # move cops
env.render()
env.step([4])   # move rob
env.render()