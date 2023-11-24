from models import betaChessAI, valueNet
from env import ChessEnvV1
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--batch-size", type=int, default=64)

args = parser.parse_args()
chessModel = betaChessAI()

LR = 1e-4
GAMMA = 0.99
EPOCHS = 10
CLIP_EPS = 0.2
BATCH_SIZE = 64

env = ChessEnvV1()

chessModel = betaChessAI()
valueModel = valueNet()
policy_optim = optim.Adam(chessModel.parameters(), lr=LR)
value_optim = optim.Adam(valueModel.parameters(), lr=LR)
loss = nn.MSELoss()

def compute_returns(rewards):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    return torch.tensor(returns)

def PPO_step():
    state = env.reset()
    done = False
    states, actions, log_probs_old, rewards = [], [], [], []
    
    chessModel.eval()
    valueModel.eval()
    while not done:
        action_probs = chessModel(state)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        log_probs_old.append(torch.log(action_probs[0, action])) ## why
        rewards.append(reward)

        state = next_state
    returns = compute_returns(rewards)
    values = valueModel(torch.stack(states))
    advantages = returns - values.squeeze()

    chessModel.train()
    valueModel.train()
    for _ in range(EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            batch_states = torch.stack(states[i:i+BATCH_SIZE])
            batch_actions = torch.tensor(actions[i:i+BATCH_SIZE])
            batch_log_probs_old = torch.stack(log_probs_old[i:i+BATCH_SIZE])
            batch_advantages = advantages[i:i+BATCH_SIZE]
            batch_returns = returns[i:i+BATCH_SIZE]

            new_action_probs = chessModel(batch_states)
            new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1)))
            ratio = (new_log_probs - batch_log_probs_old).exp()

            surrogate_obj1 = ratio * batch_advantages
            surrogate_obj2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages
            policy_loss = -torch.min(surrogate_obj1, surrogate_obj2).mean()

            value_loss = loss(valueModel(batch_states), batch_returns.unsqueeze(-1))

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

for _ in range(1000):
    PPO_step()