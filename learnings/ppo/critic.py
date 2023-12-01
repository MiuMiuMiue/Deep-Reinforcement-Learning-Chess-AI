import torch as T
import torch.nn as nn
from utils import build_base_model, betaChessValue


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_layers: tuple[int]) -> None:
        super().__init__()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.hidden_layers = hidden_layers
        self.model = betaChessValue().to(self.device)

    def forward(self, state: T.Tensor):
        x = self.model(state)

        return x
