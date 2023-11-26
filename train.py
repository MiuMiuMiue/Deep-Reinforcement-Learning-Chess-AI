from models import betaChessAI, valueNet
from env import ChessEnvV1
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import argparse
from copy import deepcopy
import random

parser = argparse.ArgumentParser()

parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--resume-point", type=int, default=1)
parser.add_argument("--ckpt", type=str, default="")

args = parser.parse_args()
chessModel = betaChessAI()

device = 0
LR = parser.learning_rate
GAMMA = parser.gamma
EPOCHS = parser.epochs
CLIP_EPS = parser.clip_eps
BATCH_SIZE = parser.batch_size

env = ChessEnvV1()

chessModel = betaChessAI()
valueModel = valueNet()
policy_optim = optim.Adam(chessModel.parameters(), lr=LR)
value_optim = optim.Adam(valueModel.parameters(), lr=LR)

if parser.ckpt:
    chessModel, valueModel, policy_optim, value_optim = resume_from_ckpt(chessModel, valueModel, policy_optim, value_optim, parser.ckpt)
    ema_teacher = deepcopy(chessModel).to(device)
    update_ema(ema_teacher, chessModel.module)
else:
    ema_teacher = deepcopy(chessModel).to(device)
    update_ema(ema_teacher, chessModel.module, decay=0)

players = (chessModel, ema_teacher)
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

    side = 0
    player = random.choice([0, 1])
    while not done:
        actions = env.possible_actions

        action_probs = players[player](state, actions, side)
        action = torch.multinomial(action_probs, 1).item()
        next_state, reward, done, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        log_probs_old.append(torch.log(action_probs[0, action])) ## why
        rewards.append(reward)

        state = next_state
        side = 1 - side
        player = 1 - player

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

for i in range(args.resume_point, 1001):
    PPO_step()
    if i % 10 == 0:
        save_ckpt(chessModel, valueModel, policy_optim, value_optim, args.result_dir, i)