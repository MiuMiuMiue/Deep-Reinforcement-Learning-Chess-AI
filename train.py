from chess import Chess
from agents import SingleAgentChess, DoubleAgentsChess
from learnings.ppo import PPO
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--start-episode", type=int, default=0)
parser.add_argument("--agent", type=str, default="single")
parser.add_argument("--episodes", type=int, default=1000)

args = parser.parse_args()

if args.start_episode > 0:
    args.start_episode += 1 # so it continues to sampling instead of directly going to training epochs

buffer_size = 32
if __name__ == "__main__":
    chess = Chess(window_size=512, max_steps=128, render_mode="rgb_array")
    chess.reset()

    ppo = PPO(
        chess,
        hidden_layers=(2048,) * 4,
        epochs=100,
        buffer_size=buffer_size * 2,
        batch_size=128,
    )

    print("Device:", ppo.device)

    if args.agent == "double":
        print(f"Start training Double Agents from episode {args.start_episode} ...")
        agent = DoubleAgentsChess(
            env=chess,
            learner=ppo,
            episodes=1000,
            train_on=buffer_size,
            result_folder="results/DoubleAgents",
            start_episode=args.start_episode,
            ckpt_path=args.ckpt,
        )
    else:
        print(f"Start training Single Agent from episode {args.start_episode} ...")
        agent = SingleAgentChess(
            env=chess,
            learner=ppo,
            episodes=1000,
            train_on=buffer_size,
            result_folder="results/SingleAgent",
            start_episode=args.start_episode,
            ckpt_path=args.ckpt,
        )
    agent.train(render_each=20, save_on_learn=True)
    agent.save(args.episodes)
    chess.close()
