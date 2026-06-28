"""
Training loop for the LLM-DRRN agent.

One episode looks like:

    reset()
    ┌─ for each step:
    │   act()        → LLM candidates → DRRN Q-values → biased ε-greedy
    │   env.step()
    │   observe()    → update history buffer, push to replay
    │   learn()      → TD update if buffer has enough transitions
    └─ end_episode() → decay ε

Metrics tracked:
  - score         : cumulative reward per episode
  - td_loss       : average TD loss per episode
  - steps         : steps taken per episode
  - epsilon       : exploration rate
"""

import logging
import os
import time
from collections import deque
from typing import Dict, List, Optional

from src.agent.agent import LLMDRRNAgent
from src.environment.textworld_env import TextWorldEnv

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        train_cfg = config["training"]

        self.env = TextWorldEnv(
            game_path=config["environment"]["game_path"],
            max_steps=config["environment"]["max_steps"],
        )
        self.agent = LLMDRRNAgent(config)

        self.num_episodes: int = train_cfg["num_episodes"]
        self.checkpoint_dir: str = train_cfg["checkpoint_dir"]
        self.checkpoint_freq: int = train_cfg["checkpoint_freq"]
        self.log_freq: int = train_cfg["log_freq"]

        # Rolling statistics (last 10 episodes)
        self._score_window: deque = deque(maxlen=10)
        self._best_score: float = float("-inf")

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info("=" * 60)
        logger.info("Starting training: %d episodes", self.num_episodes)
        logger.info("=" * 60)

        for episode in range(1, self.num_episodes + 1):
            metrics = self._run_episode(training=True)
            self._score_window.append(metrics["score"])

            if episode % self.log_freq == 0:
                avg_score = sum(self._score_window) / len(self._score_window)
                logger.info(
                    "Episode %4d | score=%.2f (avg10=%.2f) | steps=%2d | "
                    "loss=%.4f | ε=%.3f | replay=%d",
                    episode,
                    metrics["score"],
                    avg_score,
                    metrics["steps"],
                    metrics.get("avg_loss", 0.0),
                    metrics["epsilon"],
                    metrics["replay_size"],
                )

            # Save best checkpoint
            if metrics["score"] > self._best_score:
                self._best_score = metrics["score"]
                best_path = os.path.join(self.checkpoint_dir, "best.pt")
                self.agent.save(best_path)
                logger.info("  ↑ New best score %.2f → saved to %s", self._best_score, best_path)

            # Periodic checkpoint
            if episode % self.checkpoint_freq == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"ep_{episode:04d}.pt")
                self.agent.save(ckpt_path)

        logger.info("Training complete. Best score: %.2f", self._best_score)
        self.env.close()

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    def _run_episode(self, training: bool = True) -> Dict:
        obs, infos = self.env.reset()
        self.agent.reset(obs)

        admissible = self.env.get_admissible_commands(infos)

        total_score: float = 0.0
        losses: List[float] = []
        step: int = 0
        done: bool = False

        while not done:
            # ---- Select action (Modules 1 + 2 + 3 + 4) ----
            action = self.agent.act(admissible, training=training)

            # ---- Step environment ----
            next_obs, reward, done, infos = self.env.step(action)
            next_admissible = self.env.get_admissible_commands(infos)
            total_score += reward
            step += 1

            # ---- Store transition (Module 1 update + replay push) ----
            self.agent.observe(
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                next_admissible=next_admissible,
            )

            # ---- TD update (Module 5) ----
            loss = self.agent.learn()
            if loss is not None:
                losses.append(loss)

            # Advance to next step
            admissible = next_admissible

        self.agent.end_episode()

        return {
            "score": total_score,
            "steps": step,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "epsilon": self.agent.policy.current_epsilon,
            "replay_size": len(self.agent.replay),
        }

    # ------------------------------------------------------------------
    # Evaluation (greedy, no training)
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict:
        n = num_episodes or self.cfg["evaluation"]["num_episodes"]
        scores: List[float] = []
        steps: List[int] = []

        logger.info("Evaluating for %d episodes (greedy)...", n)
        for _ in range(n):
            metrics = self._run_episode(training=False)
            scores.append(metrics["score"])
            steps.append(metrics["steps"])

        avg_score = sum(scores) / len(scores)
        avg_steps = sum(steps) / len(steps)
        logger.info(
            "Eval result: avg_score=%.3f  avg_steps=%.1f  max=%.2f  min=%.2f",
            avg_score, avg_steps, max(scores), min(scores),
        )
        return {"avg_score": avg_score, "avg_steps": avg_steps, "scores": scores}
