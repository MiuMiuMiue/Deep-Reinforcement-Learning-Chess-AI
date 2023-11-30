import torch as T
import torch.nn as nn
from utils import build_base_model, betaChessValue


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.model = build_base_model(state_dim, hidden_layers, 1)
        self.model = betaChessValue()

    def forward(self, state: T.Tensor):
        x = self.model(state)

        return x
