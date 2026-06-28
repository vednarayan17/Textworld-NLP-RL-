"""
Evaluation entry point.

Loads a saved checkpoint and runs the agent in greedy mode (ε=0) for N
episodes, reporting average score, steps, and per-episode breakdown.

Usage:
    python evaluate.py                                   # uses config defaults
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --episodes 50
    python evaluate.py --backend mock                   # no-LLM eval
"""

import argparse
import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LLM-DRRN agent.")
    parser.add_argument("--config",     default="config.yaml",             help="Config YAML path.")
    parser.add_argument("--checkpoint", default=None,                      help="Checkpoint .pt file.")
    parser.add_argument("--episodes",   type=int, default=None,            help="Number of eval episodes.")
    parser.add_argument("--backend",    default=None,                      help="Override LLM backend.")
    parser.add_argument("--game",       default=None,                      help="Override game path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.backend:
        config["llm"]["backend"] = args.backend
    if args.game:
        config["environment"]["game_path"] = args.game

    checkpoint = args.checkpoint or config["evaluation"]["checkpoint_path"]
    num_episodes = args.episodes or config["evaluation"]["num_episodes"]

    if not os.path.exists(checkpoint):
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    from src.training.trainer import Trainer

    trainer = Trainer(config)
    trainer.agent.load(checkpoint)
    # Force greedy evaluation
    trainer.agent.policy.epsilon = 0.0

    results = trainer.evaluate(num_episodes=num_episodes)

    print("\n" + "=" * 50)
    print(f"  Avg Score : {results['avg_score']:.4f}")
    print(f"  Avg Steps : {results['avg_steps']:.1f}")
    print(f"  Max Score : {max(results['scores']):.4f}")
    print(f"  Min Score : {min(results['scores']):.4f}")
    print("=" * 50)

    trainer.env.close()


if __name__ == "__main__":
    main()
