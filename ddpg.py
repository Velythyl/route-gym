import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


def make_base_layers(state_dim, action_dim):
    modules = []
    old_dim = state_dim
    next_dim = state_dim // 2
    while next_dim >= 2 * action_dim:
        modules.append(nn.Linear(old_dim, next_dim))
        modules.append(nn.ReLU())
        old_dim = next_dim
        next_dim //= 2
    return modules, old_dim

class ActorDense(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorDense, self).__init__()
        modules, next_dim = make_base_layers(state_dim, action_dim)
        self.base_modules = nn.Sequential(*modules)
        self.l1 = nn.Linear(next_dim+1+1, (next_dim+1+1)//2)
        self.softmax = nn.Softmax()

    def forward(self, x, position, goal):
        x = self.base_modules(x)
        x = self.l1(torch.cat([x,position,goal], 1))
        return self.softmax(x)


class CriticDense(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticDense, self).__init__()
        modules, next_dim = make_base_layers(state_dim, action_dim)
        self.base_modules = nn.Sequential(*modules)
        self.l1 = nn.Linear(state_dim+1+1+1, (state_dim+1+1+1)//2)
        self.l2 = nn.Linear((state_dim+1+1+1)//2, 1)

    def forward(self, x, position, goal, u):
        x = self.base_modules(x)
        x = F.relu(self.l1(torch.cat([x,position, goal, u], 1)))
        return self.l2(x)

class DDPG(object):
    def __init__(self, nb_nodes):
        super(DDPG, self).__init__()

        state_dim = nb_nodes * nb_nodes
        action_dim = nb_nodes

        self.flat = True
        self.actor = ActorDense(state_dim, action_dim).to(device)
        self.actor_target = ActorDense(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = CriticDense(state_dim, action_dim).to(device)
        self.critic_target = CriticDense(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def predict(self, state):
        """

        @param state: tuple of adjacency matrix, position, goal
        @return:
        """
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert len(state[0].shape) == 2
        mat = np.ravel(state[0])

        state = torch.IntTensor(mat).to(device)

        with torch.no_grad():
            state = state.detach()
            action = self.actor(state).cpu().data.numpy().flatten()

        return action

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):

            # Sample replay buffer
            sample = replay_buffer.sample(batch_size, flat=self.flat)
            state = torch.FloatTensor(sample["state"]).to(device)
            action = torch.FloatTensor(sample["action"]).to(device)
            next_state = torch.FloatTensor(sample["next_state"]).to(device)
            done = torch.FloatTensor(1 - sample["done"]).to(device)
            reward = torch.FloatTensor(sample["reward"]).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, filename))
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, filename), map_location=device))
        self.critic.load_state_dict(torch.load('{}/{}_critic.pth'.format(directory, filename), map_location=device))
