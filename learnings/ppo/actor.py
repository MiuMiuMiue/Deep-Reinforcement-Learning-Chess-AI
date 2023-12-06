import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from utils import build_base_model, betaChessAI
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_layers: tuple[int]
    ) -> None:
        super().__init__()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.base_model = betaChessAI().to(self.device)

    def forward(self, states: T.Tensor, action_mask: T.Tensor):
        x = self.base_model(states)
        s = action_mask.sum(dim=1)
        l = ((x * (1 - action_mask)).sum(dim=1) / s).unsqueeze(1)
        x = (x + l) * action_mask
        return Categorical(x)
