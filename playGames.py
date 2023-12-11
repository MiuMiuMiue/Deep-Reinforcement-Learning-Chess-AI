from abc import ABC, abstractmethod
import os
import numpy as np
import chess.pieces as Pieces
import chess.info_keys as InfoKeys
from learnings.ppo import PPO

from buffer.episode import Episode
from learnings.base import Learning
from tqdm import tqdm
from chess import Chess
import torch as T
from utils import save_to_video
import argparse
from agents import SingleAgentChess, DoubleAgentsChess

parser = argparse.ArgumentParser()

parser.add_argument("--white-ckpt", type=str, default="results\\DoubleAgents\\model_ckpts\\white_ppo_543.pt")
parser.add_argument("--black-ckpt", type=str, default="results\\DoubleAgents\\model_ckpts\\black_ppo_1000.pt")

args = parser.parse_args()


def loadCkpt(path, agent):
    checkpoints = T.load(path, map_location=lambda storage, loc: storage)
    policy_model = checkpoints["policyModel"]
    value_model = checkpoints["valueModel"]
    agent.actor.load_state_dict(policy_model, strict=True)
    agent.critic.load_state_dict(value_model, strict=True)
    agent.actor.eval()
    agent.critic.eval()

env = Chess(window_size=512, max_steps=128, render_mode="rgb_array")
env.reset()

whiteAgent = PPO(
        environment=env,
        hidden_layers=(1000),
        epochs=10,
        buffer_size=64,
        batch_size=64
    )

loadCkpt(args.white_ckpt, whiteAgent)
# loadCkpt("results\\DoubleAgents\\model_ckpts\\white_ppo_1000.pt", whiteAgent)

blackAgent = PPO(
        environment=env,
        hidden_layers=(1000),
        epochs=10,
        buffer_size=64,
        batch_size=64
    )

loadCkpt(args.black_ckpt, blackAgent)
# loadCkpt("results\\DoubleAgents\\model_ckpts\\black_ppo_543.pt", blackAgent)

def take_action(turn: int):
    mask = env.get_all_actions(turn)[-1]
    state = env.get_state(turn)

    if turn == 0:
        action, prob, value = blackAgent.take_action(state, mask)
    else:
        action, prob, value = whiteAgent.take_action(state, mask)

    rewards, done, infos = env.step(action)
    return done, rewards, infos

draw = 0
white = 0
black = 0

for _ in range(10):
    env.reset()

    with T.no_grad():
        while True:
            done, rewards, infos = take_action(Pieces.WHITE)
            if done:
                break

            done, rewards, infos = take_action(Pieces.BLACK)
            if done:
                break
    if rewards[0] > 0:
        black += 1
    elif rewards[1] > 0:
        white += 1
    else:
        draw += 1

print(f"Draws: {draw}")
print(f"Whites: {white}")
print(f"Blacks: {black}")