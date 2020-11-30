import numpy as np
import networkx as nx

from env import ShortestRouteEnv


def define_operators(P, R, gamma, update_globals=True):
    R = R
    nstates, nactions = P.shape[-1], P.shape[0]

    def Pd(d):
        pd = np.einsum('ast,sa->st', P, d)  # given on slack
        return pd

    def rd(d):
        return np.einsum('sa,sa->s', R, d)

    def get_policy_evaluation_operator(d):
        return lambda v: rd(d) + gamma * Pd(d) @ v

    def bellman_content(v):
        x = np.einsum("ast,t->sa", P, v)
        return R + gamma * x

    def bellman_optimality_operator(v):
        return np.max(bellman_content(v), axis=1)

    def policy_evaluation_pio(d):
        gamma_pd = gamma * Pd(d)
        gamma_pd = np.identity(nstates) - gamma_pd
        return np.linalg.solve(gamma_pd, rd(d))

    def bellman_policy(v):
        d_next = np.zeros((nstates, nactions))
        cont = bellman_content(v)
        best_actions = np.argmax(cont, axis=1)
        d_next[list(range(len(best_actions))), best_actions] = 1  # gives best action for each state
        return d_next

    def policy_iteration_operator(d):
        v = policy_evaluation_pio(d)
        return v, bellman_policy(v)

    if update_globals:
        for key, val in locals().items():
            globals()[key] = val
    return locals()


def get_pi(env):
    """
    import mdptoolbox
    from mdptoolbox.mdp import PolicyIteration
    x = PolicyIteration(env.P, env.R, 0.99999999999)
    x.run()

    pi = np.zeros((len(env.graph.ngraph.nodes),len(env.graph.ngraph.nodes)))
    pi[list(range(len(env.graph.ngraph.nodes))), x.policy] = 1

    return pi
    """


    define_operators(env.P, env.R, 0.9999999999)
    guess = np.identity(len(env.graph.ngraph.nodes))

    old_guess = None
    while True:
        if np.all(guess == old_guess):
            break

        old_guess = guess
        guess = policy_iteration_operator(guess)

        guess = guess[1].astype(int)

    return guess


if __name__ == "__main__":
    networkx_graph = nx.petersen_graph()  # DiGraph()
    # networkx_graph.add_edges_from(
    #    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
    #     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

    np.random.seed(10000000)

    env = ShortestRouteEnv(networkx_graph, 0, 3, random_weights=(1, 10))

    position = env.reset()
    env.render(human_reading_delay=1)
    define_operators(env.P, env.R, 0.99)

    initial_policy = np.identity(len(env.graph.ngraph.nodes))

    optimal_policy = get_pi(env)  # value_function-policies list

    # Let's test this out :)
    env.render(human_reading_delay=1)
    done = False
    optimal_actions = [position]
    optimal_reward = 0
    while not done:
        next_action = np.argmax(optimal_policy[position])
        optimal_actions.append(next_action)

        position, reward, done, _ = env.step(next_action)
        optimal_reward += reward

        env.render(human_reading_delay=1)

    fake_rew = 0
    R = env.R
    position = env.graph.origin
    for action in optimal_actions:
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
