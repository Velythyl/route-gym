from inspect import getmembers, isfunction

import numpy as np
import random
import networkx as nx

from routegym.env import ShortestRouteEnv, LongestRouteEnv
from routegym.train import get_policy_iteration, get_backward_induction_actions

MAKE_HORIZON = True

def env_gen(n):
    small_generators = [a for _, a in getmembers(nx.generators.small, isfunction)]

    if MAKE_HORIZON:
        for f in small_generators:
            for bad_f in ["truncated_tetrahedron_graph", "chvatal_graph",
                          "desargues_graph", "heawood_graph",
                          "dodecahedral_graph", "icosahedral_graph",
                          "pappus_graph", "moebius_kantor_graph"]:
                if bad_f in str(f):
                    small_generators.remove(f)

    for _ in range(n):
        while True:
            gen = random.choice(small_generators)
            print(gen)
            try:

                graph = gen()
                if len(graph.nodes) == 0 or len(graph.nodes) > 20:
                    raise Exception()
                break
            except:
                continue

        while True:
            o, g = np.random.randint(0, len(graph.nodes), 2)
            if o == g:
                continue
            if nx.has_path(graph, o, g):
                break

        env.reset(graph, o, g, random_weights=(0, 200), make_horizon=MAKE_HORIZON)
        yield

random.seed(2)
np.random.seed(1)
ENV_QTTY = 10000
from tqdm import trange
env = LongestRouteEnv(nx.frucht_graph(), 0, 5, random_weights=(1,10)) # Fake

RENDER = True
def render(env=env, force=False):
    if force or RENDER:
        env.render()

generator = env_gen(ENV_QTTY)
for _ in trange(ENV_QTTY):
    next(generator)

    render()
    sequence = get_backward_induction_actions(env)

    done = False
    position = env.graph.origin
    policy_path = [position]
    optimal_reward = 0

    fake_rew = 0
    for action in sequence:
        fake_rew += env.R[policy_path[-1], action]

        policy_path.append(action)
        render()

        position, reward, done, _ = env.step(action)
        optimal_reward += reward
    render()
    print("RL:", optimal_reward)
    print("OPT:", env.graph.longest_path_rew)
    assert optimal_reward == env.graph.longest_path_rew