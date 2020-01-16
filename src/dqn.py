import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, observation_space, hidden_units, outputs):
        super(DQN, self).__init__() # What does this do?
        self.dense1 = nn.Linear(observation_space,  hidden_units) # What sizes should these be?
        self.dense2 = nn.Linear(hidden_units, hidden_units)
        self.head = nn.Linear(hidden_units, outputs)
        
        
    def forward(self, x):
        x = F.relu(self.dense1(x.float()))
        x = F.relu(self.dense2(x))
        return self.head(x.view(x.size(0), -1)) # What does this do? 