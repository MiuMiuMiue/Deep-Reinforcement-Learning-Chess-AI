from env import ChessEnvV1
import random
import torch
import numpy as np

env = ChessEnvV1(log=False)

done = False

while not done:
    state = env.state

    # or select an action directly
    actions = env.possible_actions
    action = random.choice(actions)
    new_state, reward, done, info = env.step(action)
    print(new_state)
    break

B = 1
side = [1]
x = [new_state]

boards = np.zeros((B, 13, 8, 8))
for batch in range(B):
    for piece in range(1, 7):
        boards[batch][piece - 1, :, :] = x[batch] == piece

    for piece in range(1, 7):
        boards[batch][piece + 5, :, :] = x[batch] == -1 * piece

    boards[batch][12, :, :] = side[batch]

print(boards)