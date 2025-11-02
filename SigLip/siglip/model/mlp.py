import torch 
import torch.nn as nn 

class SiglipMLP(nn.Module): 
    def __init__(self, config): 
        super().__init__() 
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) 
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU(approximate="tanh") 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states) 
        hidden_states = self.fc2(hidden_states)
        return hidden_states 
