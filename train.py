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

        Q = np.empty((nactions, nstates))
        for a in range(nactions):
            Q[a] = R[a] + gamma * P[a].dot(v)
        temp = Q.argmin(axis=0)

        d_next = np.zeros((nstates, nactions))
        d_next[list(range(nstates)), temp] = 1  # gives best action for each state
        return d_next


        """
        d = np.zeros((len(v), len(v)))

        for s in range(len(v)):
            the_a = []
            for a in range(len(v)):
                Pa = P[a]
                summed = 0
                for sp in range(len(v)):
                    summed += Pa[s, sp] * v
                summed *= gamma
                summed += R[s, a]
                the_a.append(np.sum(summed))
            the_a = np.array(the_a)
            the_a = np.argmax(the_a)
            d[s,the_a] = 1
        return d
        x = R
        """
        d_next = np.zeros((nstates, nactions))
        print("a")

        cont = bellman_content(v)
        best_actions = np.argmax(cont, axis=1)
        d_next[list(range(len(best_actions))), best_actions] = 1  # gives best action for each state
        return d_next

        """
    x = R
    d_next = np.zeros((nstates, nactions))
    bell_c = bellman_content(v)

    best_actions = np.max(bell_c, axis=1)
    best_actions = (bell_c == best_actions[:,None])*1
    best_actions_sum = np.sum(best_actions, axis=1, keepdims=True)
    best_actions = best_actions / best_actions_sum

    #d_next[best_actions] = 1 # gives best action for each state
    return best_actions
    """

    def policy_iteration_operator(d):
        v = policy_evaluation_pio(d)
        return v, bellman_policy(v)

    def smooth_policy_iteration_operator(d, tau):
        v = policy_evaluation_pio(d)

        content = np.exp(bellman_content(v) / tau)
        d_next = content / sum(content)

        return v, d_next

    def newton_kantorovich_operator(v):
        return v + np.linalg.inv(np.identity(nstates) - gamma * Pd(bellman_policy(v))) @ (
                    bellman_optimality_operator(v) - v)

    if update_globals:
        for key, val in locals().items():
            globals()[key] = val
    return locals()


# Bad form for complex examples as this is writable looplessly, but it's good enough for this use case
def criter(guess, olds):
    if len(olds) == 0:
        return False
    return np.all(guess == olds[-1])
    for x in olds + [guess]:
        if np.isnan(np.sum(x)):
            return True

    for old in reversed(olds):
        if np.allclose(guess, old, equal_nan=True):
            return True
    return False


# op: the operator
# guess: the inital guess
# caller: how we're calling the operator. Useful for smooth bellman: caller=lambda op, guess: op(guess, tau). By default, caller is just op(guess)
# criterion: when to stop. By default, doesn't use a threshold, and stops when the old guess is the same as the new one
def policy_finder(op, guess, caller=lambda op, guess: op(guess), criterion=criter):
    old = None

    old_v = None
    old_guess = None
    v = np.array([1])
    while True:
        try:
            print(np.argmax(guess, axis=1))
            print(np.argmax(old_guess, axis=1))
        except:
            pass
        #print(v)
        if np.all(guess == old_guess):
        #if np.all(guess == old_guess):
            print("both eq")
            break

        old_v = v
        old_guess = guess
        guess = caller(op, guess)

        #print("g")
       # print(guess)

        v = guess[0]
        guess = guess[1].astype(int)


    return guess


def get_pi(env):

    import mdptoolbox
    from mdptoolbox.mdp import PolicyIteration
    x = PolicyIteration(env.P, env.R, 0.99999999999)
    x.run()

    pi = np.zeros((len(env.graph.ngraph.nodes),len(env.graph.ngraph.nodes)))
    pi[list(range(len(env.graph.ngraph.nodes))), x.policy] = 1

    return pi



    define_operators(env.P, env.R, 0.9999999999)
    initial_policy = np.identity(len(env.graph.ngraph.nodes))
    initial_policy = np.zeros((len(env.graph.ngraph.nodes), len(env.graph.ngraph.nodes)))
    initial_policy[:,0] = 1
    return policy_finder(policy_iteration_operator, initial_policy)  # value_function-policies list


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
