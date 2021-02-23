import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.tensor(state, device=self.device)
        action = self.model(state).argmax(-1).cpu().numpy()

        return action

    def reset(self):
        pass

