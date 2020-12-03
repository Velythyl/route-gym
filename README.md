# Route-Gym
An openai gym that allows agent to solve shortest and longest route problems.

These simples are simple for humans, and can be computed easily by hand using Dijkstra's algorithm
and its variants like A* (computation time permitting). As we work towards AGI, it is of my opinion that our complicated general-purpose
algorithms should be asked to solve these simple problems. If convergence is slow, or if the supposedly 
"general" algorithm can't solve it easily,
it is safe to say that that algorithm should be reworked.

# Quickstart

    env = ShortestRouteEnv(nx.frucht_graph(), 0, 5, random_weights=(1,10))
    env.render()    # optionnal
    done = False
    # you might want to give the adjacency matrix to the policy
    policy = ?
    rew = 0
    position = origin
    while not done:
        action = policy.predict(position)
        position, reward, done, _ = env.step(action)
        end.render()    # optionnal
        rew += reward
    print("Final reward:", rew)
    print("Dijkstra's reward:", env.graph.dijkstra_rew)
        
# What is provided in this gym?

## The environment:

The two environments are based neton [OpenAI's `gym`](https://github.com/openai/gym).

The environment can be called using `routegym.env.ShortestRouteEnv` or the equivalent for the longest route version.

The environments have a `render` function you can use to display the environment's state. In it, the blue path on the
graph's arcs represents Dijkstra's path. This only works for `ShortestRouteEnv`.

The environments receive a [`networkx`](https://github.com/networkx/networkx) graph, an origin, a goal, and random weight
boundaries (if need be) as part of their constructor. 

You can also set the `make_horizon` flag to `True` to transform the graph
into a finite-horizon problem. Be warned that this should only be used on smaller graphs: this generates a tree out
of all the possible paths the agent can take from `origin` to `goal` and merges them into a single graph. Needless to say,
the big O of this thing is enormous! This flag should only be used for toy examples.

## The Graph class

The environments use a custom graph class as a backend. A typical user should not need to interact with this class.

But this class does calculate the Dijkstra solution for the problem. If you want to compare your algorithm's performance
to Dijkstra's, you can use `env.graph.dijkstra_path` to get Dijkstra's path, or `env.graph.dijkstra_rew` to get Dijkstra's reward.

## The validate script

This is an internal test, but it is provided as a courtesy to the users. You can get inspiration from that script, either
as a tutorial on how to use this package, or as a way to generate many environments, etc.

You can view it in `routegym.validate.py`.