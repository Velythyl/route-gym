import numpy as np
import networkx as nx

from env import ShortestRouteEnv


def define_operators(P, R, gamma, update_globals=True):
  nstates, nactions = P.shape[-1], P.shape[0]

  def Pd(d):
    pd = np.einsum('ast,sa->st', P, d)  # given on slack
    return pd

  def rd(d):
    return np.einsum('sa,sa->s', R, d)

  def bellman_content(v): # sneaky transposed normal R
    return R+gamma*P@v

  def bellman_optimality_operator(v):
    return np.max(bellman_content(v), axis=1)

  def policy_evaluation_pio(d):
    gamma_pd = np.identity(nstates)-gamma*Pd(d)
    gamma_pd_inv = np.linalg.inv(gamma_pd)
    v = gamma_pd_inv @ rd(d)
    return v

  def bellman_policy(v):
    """
      def bellman_policy(v):
        x = R
        d_next = np.zeros((nstates, nactions))

        best_actions = np.argmax(bellman_content(v), axis=1)
        d_next[list(range(len(best_actions))), best_actions] = 1 # gives best action for each state
        return d_next
    """

    x = R
    d_next = np.zeros((nstates, nactions))
    bell_c = bellman_content(v)

    best_actions = np.max(bell_c, axis=1)
    best_actions = bell_c == best_actions[:,None]

    d_next[best_actions] = 1 # gives best action for each state
    return d_next



  def policy_iteration_operator(d):
    v = policy_evaluation_pio(d)
    return v, bellman_policy(v)

  def smooth_policy_iteration_operator(d, tau):
    v = policy_evaluation_pio(d)

    content = np.exp(bellman_content(v)/tau)
    d_next = content/sum(content)

    return v, d_next

  def newton_kantorovich_operator(v):
    return v + np.linalg.inv(np.identity(nstates) - gamma * Pd(bellman_policy(v)))@(bellman_optimality_operator(v) - v)

  if update_globals:
    for key, val in locals().items():
      globals()[key] = val
  return locals()


# Bad form for complex examples as this is writable looplessly, but it's good enough for this use case
def criter(guess, olds):
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
  if len(guess.shape) == 1:
    yield guess

  olds = []

  if criterion is None:
    criterion = criter

  while not criterion(guess, olds):

    print(guess)
    olds.append(guess)
    guess = caller(op, guess)

    if isinstance(guess, tuple):
      v = guess[0]
      guess = guess[1]
    else:
      v = guess

    yield v, guess

networkx_graph = nx.DiGraph()
networkx_graph.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
env = ShortestRouteEnv(networkx_graph, 0, 5)

position = env.reset()
env.render(human_reading_delay=1)
define_operators(env.P, env.R, 0.9)

initial_policy = np.identity(len(env.graph.ngraph.nodes))

vp_s = list(policy_finder(policy_iteration_operator, initial_policy)) # value_function-policies list
optimal_policy = vp_s[-1][1]

# Let's test this out :)
env.render(human_reading_delay=2)
done = False
optimal_actions = [position]
optimal_reward = 0
while not done:
  next_action = np.argmax(optimal_policy[position])
  optimal_actions.append(next_action)

  position, reward, done, _ = env.step(next_action)
  optimal_reward += reward

  env.render(human_reading_delay=2)

# Do we have Dijkstra?
print("Dijkstra's path: ", env.graph.dijkstra_path,
      "Our path: ", optimal_actions,
      "Is equal?: ", np.all(np.array(env.graph.dijkstra_path) == np.array(optimal_actions))
)

print("Disjktra's reward: ", env.graph.dijkstra_rew, "Our reward: ", optimal_reward)