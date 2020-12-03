import numpy as np


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

def get_policy_iteration(env, use_mdptoolbox=False):
    if use_mdptoolbox:
        from mdptoolbox.mdp import PolicyIteration
        x = PolicyIteration(env.P, env.R, 0.99999999999)
        x.run()

        pi = np.zeros((len(env.graph.ngraph.nodes), len(env.graph.ngraph.nodes)))
        pi[list(range(len(env.graph.ngraph.nodes))), x.policy] = 1

        return pi
    else:
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

def get_backwards_induction_policy(env):
    max_horizon = len(env.graph.horizon_states)