from env import ChessEnvV1
import random
import torch.nn as nn
import numpy as np
import torch
from models import betaChessAI
from copy import deepcopy
from utils import *

# env = ChessEnvV1(log=False, opponent="none")
# student = betaChessAI()
# teacher = deepcopy(student)
# update_ema(teacher, student)

# requires_grad(teacher, False)
# done = False

# side = random.choice((0, 1))
# print(side)
# count = 0
# state = env.reset(opponent=teacher)

# while not done:
#     print("=", end="")
#     actions = env.possible_actions

#     action_probs = student(torch.tensor([state]), torch.tensor([actions]), torch.tensor([side]))[0]
#     action = torch.multinomial(action_probs, 1).item()
#     while action not in actions:
#         action = torch.multinomial(action_probs, 1).item()
#     new_state, reward, done, info = env.step(action)

#     # count += 1
#     # if count == 1:
#     #     break

# print()
# print(new_state)
# print(reward)
legal_actions = torch.tensor([134, 431, 325])
test = nn.functional.pad(legal_actions, (0, 10 - len(legal_actions)), value=-1)
print(test)
