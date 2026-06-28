"""
TextWorld game generator.

Run this once before training to create the .ulx game file that the agent
learns to play. Several generation strategies are tried in order:

  1. tw-make CLI tool  (simplest, always works if TextWorld is installed)
  2. textworld.challenges coin-collector Python API
  3. textworld.GameOptions Python API (fallback)

Usage:
    python generate_game.py [--output games/] [--seed 42] [--rooms 5] [--coins 3]
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def generate_via_cli(output_dir: str, seed: int, rooms: int, quest_length: int) -> str:
    """Use the tw-make command-line tool (fastest, most reliable)."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "tw-make", "custom",
        "--world-size", str(rooms),
        "--nb-objects", str(rooms * 2),
        "--quest-length", str(quest_length),
        "--seed", str(seed),
        "--output", output_dir,
        "--silent",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"tw-make failed:\n{result.stderr}")

    # tw-make writes the .ulx file into output_dir; find it
    ulx_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".ulx")
    ]
    if not ulx_files:
        raise FileNotFoundError(f"No .ulx file found in {output_dir}")
    return ulx_files[-1]


def generate_via_coin_collector(output_dir: str, seed: int, rooms: int, nb_coins: int) -> str:
    """Use TextWorld's built-in coin-collector challenge."""
    import textworld.challenges.coin_collector as tw_cc
    import textworld

    options = textworld.GameOptions()
    options.seeds = seed
    options.nb_rooms = rooms

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating coin-collector game (seed=%d, rooms=%d, coins=%d)...", seed, rooms, nb_coins)

    # The coin collector challenge has its own options
    game_options = tw_cc.CoinCollectorOptions(nb_coins=nb_coins)
    game_options.seeds = seed

    game_file, _ = tw_cc.make(game_options, path=output_dir)
    return game_file


def generate_via_game_options(output_dir: str, seed: int, rooms: int) -> str:
    """Fallback: use raw textworld.make() with GameOptions."""
    import textworld

    options = textworld.GameOptions()
    options.seeds = seed
    options.nb_rooms = rooms
    options.nb_objects = rooms * 2
    options.quest_length = max(3, rooms - 1)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating game with GameOptions (seed=%d, rooms=%d)...", seed, rooms)

    game = textworld.make(options)
    game_file = game.save(output_dir)
    return game_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TextWorld game for training.")
    parser.add_argument("--output", default="games/", help="Output directory for game files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--rooms", type=int, default=5, help="Number of rooms.")
    parser.add_argument("--coins", type=int, default=3, help="Number of coins (coin-collector only).")
    parser.add_argument("--quest-length", type=int, default=5, help="Quest length (tw-make).")
    args = parser.parse_args()

    strategies = [
        ("tw-make CLI",          lambda: generate_via_cli(args.output, args.seed, args.rooms, args.quest_length)),
        ("coin-collector API",   lambda: generate_via_coin_collector(args.output, args.seed, args.rooms, args.coins)),
        ("GameOptions fallback", lambda: generate_via_game_options(args.output, args.seed, args.rooms)),
    ]

    game_file = None
    for name, fn in strategies:
        try:
            game_file = fn()
            logger.info("Game generated via %-25s  →  %s", name, game_file)
            break
        except Exception as exc:
            logger.warning("Strategy '%s' failed: %s", name, exc)

    if game_file is None:
        logger.error("All generation strategies failed. Is TextWorld installed?")
        logger.error("  pip install textworld")
        sys.exit(1)

    logger.info("")
    logger.info("Success! Update config.yaml:")
    logger.info('  environment:')
    logger.info('    game_path: "%s"', game_file)


if __name__ == "__main__":
    main()
