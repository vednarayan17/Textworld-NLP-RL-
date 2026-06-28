"""
Training entry point for the LLM-DRRN TextWorld agent.

Usage:
    python train.py                           # uses config.yaml defaults
    python train.py --config config.yaml      # explicit config path
    python train.py --backend mock            # no-LLM ablation run
    python train.py --episodes 200            # override num_episodes
    python train.py --resume checkpoints/best.pt
"""

import argparse
import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM-DRRN agent on TextWorld.")
    parser.add_argument("--config",   default="config.yaml",  help="Path to config YAML.")
    parser.add_argument("--backend",  default=None,           help="Override LLM backend (ollama|huggingface|mock).")
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes.")
    parser.add_argument("--resume",   default=None,           help="Checkpoint path to resume from.")
    parser.add_argument("--game",     default=None,           help="Override game_path in config.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)

    # Command-line overrides
    if args.backend:
        config["llm"]["backend"] = args.backend
        logger.info("LLM backend overridden to: %s", args.backend)
    if args.episodes:
        config["training"]["num_episodes"] = args.episodes
    if args.game:
        config["environment"]["game_path"] = args.game

    game_path = config["environment"]["game_path"]
    if not os.path.exists(game_path):
        logger.error(
            "Game file not found: %s\n"
            "Run first:  python generate_game.py --output games/\n"
            "Then update config.yaml → environment.game_path",
            game_path,
        )
        sys.exit(1)

    # Lazy import so startup errors are clean
    from src.training.trainer import Trainer

    trainer = Trainer(config)

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.agent.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
