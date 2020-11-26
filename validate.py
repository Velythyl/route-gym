from inspect import getmembers, isfunction

import numpy as np
import random
import networkx as nx

from env import ShortestRouteEnv
from train import get_pi


def env_gen(n):
    small_generators = [nx.frucht_graph]#[a for _, a in getmembers(nx.generators.small, isfunction)]
    classic_generators = [a for _, a in getmembers(nx.generators.classic, isfunction)] + [a for _, a in getmembers(nx.generators.lattice, isfunction)]
    sto_generators = [nx.fast_gnp_random_graph, nx.erdos_renyi_graph, nx.binomial_graph]

    gens = [small_generators, classic_generators, sto_generators]

    for _ in range(n):
        gen_list = random.choice(gens)
        nb_nodes = np.random.randint(10,20)
        while True:
            gen = random.choice(small_generators)
            try:
                graph = gen()
                break
            except:
                try:
                    graph = gen(nb_nodes)
                    break
                except:
                    try:
                        graph = gen(nb_nodes, 0.85)
                        break
                    except:
                        pass

        while True:
            o, g = np.random.randint(0, len(graph.nodes), 2)
            if o == g:
                continue
            if nx.has_path(graph, o, g):
                break

        env.reset(graph, o, g, random_weights=(1, 200), make_horizon=True)
        yield
#np.random.seed(1)
np.random.seed(100)


ENV_QTTY = 1000
from tqdm import trange
env = ShortestRouteEnv(nx.frucht_graph(), 0, 5, random_weights=(1,10)) # Fake
#env.render()
#env.graph._define_finite_problem()

RENDER = True
def render(env=env, force=False):
    if force or RENDER:
        env.render()

generator = env_gen(ENV_QTTY)
for _ in trange(ENV_QTTY):
    next(generator)

    render()
    policy = get_pi(env)

    done = False
    position = env.graph.origin
    policy_path = [position]
    optimal_reward = 0

    fake_rew = 0
    while not done:
        next_action = np.argmax(policy[position])

        fake_rew += env.R[policy_path[-1], next_action]
        if next_action in policy_path:
            print("AH")
            i=0

        policy_path.append(next_action)
        render()

        position, reward, done, _ = env.step(next_action)
        optimal_reward += reward
        print("O:", optimal_reward, "F:", fake_rew)
    render()
    #env.close()

    print("P:", optimal_reward, "D:", env.graph.dijkstra_rew)
    print("P:", policy_path, "D:", env.graph.dijkstra_path)

    fake_rew = 0
    R = env.R
    position = env.graph.origin
    for action in policy_path:
        if action == env.graph.origin:
            continue
        fake_rew += R[position, action]
        position = action

    R = env.R
    position = env.graph.origin
    fake_rew2 = 0
    for action in env.graph.dijkstra_path:
        if action == env.graph.origin:
            continue
        fake_rew2 += R[position, action]
        position = action

    print("Disjktra's reward: ", env.graph.dijkstra_rew, "Our reward: ", optimal_reward)

    print("What RL thinks Diksjta does: ", fake_rew2, "What RL thinks it got: ", fake_rew)
    if optimal_reward != env.graph.dijkstra_rew:
        pass
    assert optimal_reward == env.graph.dijkstra_rew

311
298