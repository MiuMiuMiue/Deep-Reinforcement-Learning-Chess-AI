from env.chess_v1 import ChessEnvV1
from gym.envs.registration import register

register(
    id="ChessVsRandomBot-v1",
    entry_point="gym_chess.envs:ChessEnvV1",
    kwargs={"opponent": "random"},
)

register(
    id="ChessVsSelf-v1",
    entry_point="gym_chess.envs:ChessEnvV1",
    kwargs={"opponent": "none"},
)