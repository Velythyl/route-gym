import time

import networkx as nx

from controller import Optimal
from env import ShortestRouteEnv
from graph import Graph

networkx_graph = nx.frucht_graph()
env = ShortestRouteEnv(networkx_graph, 0, 10)

print("A")
env.render(human_reading_delay=1)
env.reset(origin=0, goal=5)
controller = Optimal(env)
done = False
while not done:
    pred = controller.predict(env.graph.position)
    obs, rew, done, _ = env.step(pred)
    print(obs)
    env.render(human_reading_delay=1)
from time import sleep
exit()