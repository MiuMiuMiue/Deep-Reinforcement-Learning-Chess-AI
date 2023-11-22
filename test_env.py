from env import ChessEnvV1
import random
import torch.nn as nn
import numpy as np
import torch

env = ChessEnvV1(log=False)

done = False

while not done:
    state = env.state

    # or select an action directly
    actions = env.possible_actions
    action = random.choice(actions)
    new_state, reward, done, info = env.step(action)
    print(new_state)
    print(actions)
    break

# batch_size = 2
# # matrix = np.random.random((batch_size, 69, 8, 8)) # example data
# matrix = np.zeros((batch_size, 69, 8, 8)) # example data

# matrix[0, 2, 1, 0] = 0.9
# matrix[0, 65, 7, 7] = 0.5
# matrix[0, 45, 6, 4] = 0.3

# matrix[1, 2, 1, 0] = 0.67
# matrix[1, 65, 5, 7] = 0.9
# matrix[1, 45, 3, 4] = 0.3

# # Reshape the matrix for softmax
# reshaped_matrix = matrix.reshape(batch_size, -1)

# sm = nn.Softmax(dim=1)

# # Apply softmax to each batch
# # softmaxed = np.exp(reshaped_matrix) / np.sum(np.exp(reshaped_matrix), axis=1, keepdims=True)
# softmaxed = sm(torch.tensor(reshaped_matrix))

# # Find the index of the max value in each batch
# max_indices = np.argmax(softmaxed, axis=1)

# # Convert flat indices back to 3D indices
# max_indices_3d = [(idx // (8*8), (idx % (8*8)) // 8, idx % 8) for idx in max_indices]

# print(max_indices_3d)  # This will give you the 3D indices in the format (plane, row, column)