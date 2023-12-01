from copy import deepcopy
from chess import Chess
from .base import BaseAgent
from buffer.episode import Episode
from learnings.base import Learning
import os


class DoubleAgentsChess(BaseAgent):
    def __init__(
        self,
        env: Chess,
        learner: Learning,
        episodes: int,
        train_on: int,
        result_folder: str,
        start_episode: int = 0, 
        ckpt_path: str = ""
    ) -> None:
        super().__init__(env, learner, episodes, train_on, result_folder, start_episode)
        self.white_agent = deepcopy(learner)
        self.black_agent = deepcopy(learner)

        if ckpt_path != "":
            self.loadCkpt(os.path.join(ckpt_path, f"black_ppo_{start_episode - 1}.pt"), self.black_agent)
            self.loadCkpt(os.path.join(ckpt_path, f"white_ppo_{start_episode - 1}.pt"), self.white_agent)
            print("Finish Loading checkpoints")

    def add_episodes(self, white: Episode, black: Episode) -> None:
        self.white_agent.remember(white)
        self.black_agent.remember(black)

    def learn(self):
        self.white_agent.learn()
        self.black_agent.learn()

    def save_learners(self, ep):
        self.white_agent.save(self.result_folder, "white_ppo", ep)
        self.black_agent.save(self.result_folder, "black_ppo", ep)
        
