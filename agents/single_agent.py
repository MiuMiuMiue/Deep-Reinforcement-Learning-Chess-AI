from chess import Chess

from .base import BaseAgent
from buffer.episode import Episode
from learnings.base import Learning
import os


class SingleAgentChess(BaseAgent):
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
        if ckpt_path != "":
            self.loadCkpt(os.path.join(ckpt_path, f"single_agent_{start_episode - 1}.pt"), self.learner)
            print("Finish Loading checkpoints")

    def add_episodes(self, white: Episode, black: Episode) -> None:
        self.learner.remember(white)
        self.learner.remember(black)

    def learn(self):
        self.learner.learn()

    def save_learners(self, ep):
        self.learner.save(self.result_folder, "single_agent", ep)
