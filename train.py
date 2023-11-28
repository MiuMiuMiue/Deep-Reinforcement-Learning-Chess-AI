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
from tqdm import tqdm
import time

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

LR = args.learning_rate
GAMMA = args.gamma
EPOCHS = args.epochs
CLIP_EPS = args.clip_eps
BATCH_SIZE = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ChessEnvV1(log=False, device=device)

chessModel = betaChessAI(device=device).to(device)
valueModel = valueNet(device=device).to(device)
policy_optim = optim.Adam(chessModel.parameters(), lr=LR)
value_optim = optim.Adam(valueModel.parameters(), lr=LR)

if args.ckpt:
    chessModel, valueModel, policy_optim, value_optim = resume_from_ckpt(chessModel, valueModel, policy_optim, value_optim, args.ckpt)
    ema_teacher = deepcopy(chessModel).to(device)
    update_ema(ema_teacher, chessModel)
else:
    ema_teacher = deepcopy(chessModel).to(device)
    update_ema(ema_teacher, chessModel, decay=0)

student = deepcopy(chessModel).to(device)
requires_grad(student, False)
requires_grad(ema_teacher, False)
student.eval()
ema_teacher.eval()
loss = nn.MSELoss()

def compute_returns(rewards):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)

    return torch.tensor(returns)

def PPO_step():
    with torch.no_grad():
        states, actions, log_probs_old, rewards = [], [], [], []
        sides, all_legal_actions = [], []
        for _ in range(5):
            side = random.choice((0, 1))
            state = env.reset(player_color="WHITE", opponent=ema_teacher) if side == 0 else env.reset(player_color="BLACK", opponent=ema_teacher)
            done = False
            while not done:
                legal_actions = env.possible_actions
                action_probs = student(torch.tensor([state]).to(device), torch.tensor([legal_actions]).to(device), torch.tensor([side]).to(device))
                action = torch.multinomial(action_probs, 1).item()
                while action not in legal_actions:
                    action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _ = env.step(action)
                
                states.append(torch.tensor(state))
                actions.append(action)
                log_probs_old.append(torch.log(action_probs[0, action]))
                rewards.append(reward)

                sides.append(torch.tensor(side))
                legal_actions = torch.tensor(legal_actions)
                all_legal_actions.append(nn.functional.pad(legal_actions, (0, 50 - len(legal_actions)), value=-1))

                state = next_state
    returns = compute_returns(rewards).to(device)
    values = valueModel(torch.stack(states).to(device), torch.stack(sides).to(device))
    advantages = returns - values.squeeze()

    for _ in range(EPOCHS):
        for i in range(0, len(states), BATCH_SIZE):
            # print("===")
            batch_states = torch.stack(states[i:i+BATCH_SIZE]).to(device)
            batch_actions = torch.tensor(actions[i:i+BATCH_SIZE]).to(device)
            batch_log_probs_old = torch.stack(log_probs_old[i:i+BATCH_SIZE]).to(device)
            batch_advantages = advantages[i:i+BATCH_SIZE].to(device)
            batch_returns = returns[i:i+BATCH_SIZE].to(device)
            batch_sides = torch.tensor(sides[i:i+BATCH_SIZE]).to(device)
            batch_legal_actions = torch.stack(all_legal_actions[i:i+BATCH_SIZE]).to(device)

            # print(f"batch_states: {batch_states.shape}, batch_legal_actions: {batch_legal_actions.shape}, batch_sides: {batch_sides.shape}")
            new_action_probs = chessModel(batch_states, batch_legal_actions, batch_sides)
            # print(f"new_action_probs: {new_action_probs.shape}")
            new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(-1))).squeeze(1)
            # print(f"new_log_probs: {new_log_probs.shape}")
            # print(f"batch_log_probs_old: {batch_log_probs_old.shape}")
            ratio = (new_log_probs - batch_log_probs_old).exp()
            # print(f"ratio: {ratio.shape}")
            # print(f"batch_advantages: {batch_advantages.shape}")

            surrogate_obj1 = ratio * batch_advantages.detach()
            # print(surrogate_obj1)
            surrogate_obj2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * batch_advantages.detach()
            # print(surrogate_obj2)
            policy_loss = -torch.min(surrogate_obj1, surrogate_obj2).mean()
            print(f"\tpolicy_loss: {policy_loss}")
            value_loss = loss(valueModel(batch_states, batch_sides), batch_returns.unsqueeze(-1))
            print(f"\tvalue_loss: {value_loss}")
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

print("start training ...")
for i in range(args.resume_point, 1001):
    start = time.time()
    print(f"Episodes: {i}")
    PPO_step()
    if i % 10 == 0:
        save_ckpt(chessModel, valueModel, policy_optim, value_optim, args.results_dir, i)
    if i % 5 == 0:
        print("\tPlaying against teacher")
        switch = switchTeacherStudent(chessModel, ema_teacher, device)
        if switch > 5:
            student = deepcopy(chessModel)
            ema_teacher = update_ema(ema_teacher, chessModel)
            requires_grad(ema_teacher, False)
            requires_grad(student, False)
            student.eval()
            ema_teacher.eval()
            print(f"\tWin!!! Update Teacher Model. {switch} / 10")
        print(f"\tNot that Good. {switch} / 10")
    print(f"Finish episodes {i}. Using {(time.time() - start) / 60:.2f} minutes")

print("Finish Training")