from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 4096
TRANSITIONS = 500000
STEPS_PER_UPDATE = 1
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )
        self.update_target_network()

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

        self.position = 0
        self.state_buffer = torch.empty((INITIAL_STEPS, state_dim), dtype=torch.float, device=self.device)
        self.next_state_buffer = torch.empty((INITIAL_STEPS, state_dim), dtype=torch.float, device=self.device)
        self.action_buffer = torch.empty((INITIAL_STEPS, 1), dtype=torch.long, device=self.device)
        self.reward_buffer = torch.empty((INITIAL_STEPS, 1), dtype=torch.float, device=self.device)
        self.done_buffer = torch.empty((INITIAL_STEPS, 1), dtype=torch.bool, device=self.device)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically

        state, action, next_state, reward, done = transition
        
        self.state_buffer[self.position] = torch.tensor(state, device=self.device)
        self.next_state_buffer[self.position] = torch.tensor(next_state, device=self.device)
        self.action_buffer[self.position] = torch.tensor(action, device=self.device)
        self.reward_buffer[self.position] = reward
        self.done_buffer[self.position] = done
        
        self.position = (self.position + 1) % INITIAL_STEPS

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster

        batch_idx = np.random.choice(INITIAL_STEPS, BATCH_SIZE, replace=False)
        return self.state_buffer[batch_idx], self.action_buffer[batch_idx], \
                self.next_state_buffer[batch_idx], self.reward_buffer[batch_idx], \
                self.done_buffer[batch_idx]
        
    def train_step(self, batch):
        # Use batch to update DQN's network.

        state, action, next_state, reward, done = batch
        
        row_range = torch.arange(state.shape[0]).reshape(-1, 1)
        with torch.no_grad():
            q_t = torch.max(self.target_model(next_state), dim=1, keepdim=True)[0]
            q_t[done] = 0

        q_t = reward + q_t * GAMMA
        q = self.model(state)[row_range, action]
        loss = F.mse_loss(q, q_t)
        
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-5, 5)

        self.optimizer.step()
        
    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.tensor(state, device=self.device)
        action = self.model(state).argmax(-1).cpu().numpy()

        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    state = env.reset()
    eps = np.linspace(0.3, 0, TRANSITIONS)
    max_reward = -100000
    
    for _ in range(INITIAL_STEPS):
        steps = 0
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        steps = 0
        
        #Epsilon-greedy policy
        if random.random() < eps[i]:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 25)
            mean_reward = np.mean(rewards)

            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            
            if mean_reward > max_reward:
                max_reward = mean_reward
                dqn.save()

