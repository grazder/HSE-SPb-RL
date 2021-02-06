from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random

SEED = 1243
random.seed(SEED)
np.random.seed(SEED)

GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30


# Simple discretization 
def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X - 1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y - 1)
    return x + GRID_SIZE_X * y


class QLearning:
    def __init__(self, state_dim, action_dim, env):
        self.qlearning_estimate = np.zeros((state_dim, action_dim)) + 2.
        self.env = env
        self.best_score = -201
        self.update_consts(0)

    def get_epsilon(self, t, min_epsilon, divisor=25):
        return max(min_epsilon, min(1, 1.0 - np.log10((t + 1) / divisor)))

    def get_lr(self, t, min_alpha, divisor=25):
        return max(min_alpha, min(1.0, 1.0 - np.log10((t + 1) / divisor)))

    def update_consts(self, i, min_eps=0.1, min_lr=0.1):
        self.eps = self.get_epsilon(i, min_eps)
        self.lr = self.get_lr(i, min_lr)

    def choose_action(self, state):
        if random.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = self.act(state)

        return action

    def update(self, transition):
        state, action, next_state, reward, done = transition

        next_action = self.choose_action(state)
        next_reward = self.qlearning_estimate[next_state][next_action] if not done else 0

        self.qlearning_estimate[state][action] = (1 - self.lr) * self.qlearning_estimate[state][action] + \
                                                 self.lr * (reward + GAMMA * next_reward)

    def act(self, state):
        return np.argmax(self.qlearning_estimate[state])

    def save(self, path, score):
        if score > self.best_score:
            self.best_score = score
            np.save(path, self.qlearning_estimate)


def evaluate_policy(agent, episodes=5):
    env = make("MountainCar-v0")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(transform_state(state)))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("MountainCar-v0")
    transitions = 4000000
    trajectory = []

    env.seed(SEED)
    env.action_space.seed(SEED)

    ql = QLearning(state_dim=GRID_SIZE_X * GRID_SIZE_Y, action_dim=3, env=env)
    state = transform_state(env.reset())

    for i in range(transitions):
        total_reward = 0
        steps = 0

        ql.update_consts(i)

        action = ql.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        reward += abs(next_state[1]) / 0.07
        next_state = transform_state(next_state)

        trajectory.append((state, action, next_state, reward, done))

        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []

        state = next_state if not done else transform_state(env.reset())

        if (i + 1) % (transitions // 100) == 0:
            rewards = evaluate_policy(ql, 25)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            ql.save('agent', np.mean(rewards))

    print('Best Score: ', ql.best_score)
