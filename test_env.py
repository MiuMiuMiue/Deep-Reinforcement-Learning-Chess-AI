from env import ChessEnvV1
import random

env = ChessEnvV1(log=False)

done = False

while not done:
    state = env.state

    # or select an action directly
    actions = env.possible_actions
    action = random.choice(actions)

    new_state, reward, done, info = env.step(action)

    print(reward)