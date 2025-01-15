import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class DQL(nn.Module):
    def __init__(self, LearningRate=0.01, Epsilon=0.3, num_layers=423):

        super().__init__()
        # Слои
        self.inp = nn.Linear(num_layers, 1024)
        self.hidden1 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 5)

        # Память
        self.memory_states = []
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_next_states = []
        self.memory_isdones = []
        self.memory_len = 0
        self.loses = []
        self.rewards_per_epoch = []

        self.optimizer = optim.Adam(self.parameters(), lr=LearningRate)
        self.criterion = nn.MSELoss()
        self.epsilon = Epsilon
        self.activation = nn.LeakyReLU()
    
    def forward(self, x):
        x = t.sigmoid(self.inp(x))
        x = t.sigmoid(self.hidden1(x))
        x = self.activation(self.out(x))
        return x
    
    def remember(self, state, action, reward, next_state, isDone):
        self.memory_states.append(state)
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_next_states.append(next_state)
        self.memory_isdones.append(1 - isDone)
        self.memory_len += 1

    # метод для получения тензоров
    def samplebatch(self):
        states = t.FloatTensor(self.memory_states[0]).cpu()
        self.memory_states.clear()

        actions = t.IntTensor(self.memory_actions).cpu()
        self.memory_actions.clear()

        rewards = t.FloatTensor(self.memory_rewards).cpu()
        self.memory_rewards.clear()

        next_states = t.FloatTensor(self.memory_next_states[0]).cpu()
        self.memory_next_states.clear()

        isdones = t.IntTensor(self.memory_isdones).cpu()
        self.memory_isdones.clear()

        self.memory_len = 0
        return (states, actions, rewards, next_states, isdones)
    
