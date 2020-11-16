import numpy as np

class Optimal:
    def __init__(self, env):
        self.env = env
    def predict(self, position):
        print(self.env.graph.dijkstra_path)
        return self.env.graph.dijkstra_path[self.env.graph.dijkstra_path.index(position)+1]
