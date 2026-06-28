"""
TextWorld gym environment wrapper.

Requests description, inventory, and admissible_commands at each step.
The admissible_commands list is the ground-truth set of valid actions —
we pass it to the LLM engine which filters it to K high-value candidates.
"""

import logging
from typing import Tuple, Dict, Any, List, Optional

import gym
import textworld.gym
from textworld import EnvInfos

logger = logging.getLogger(__name__)


class TextWorldEnv:
    def __init__(self, game_path: str, max_steps: int = 50):
        self.game_path = game_path
        self.max_steps = max_steps
        self._current_step = 0

        request_infos = EnvInfos(
            description=True,
            inventory=True,
            admissible_commands=True,
            last_command=True,
        )

        env_id = textworld.gym.register_game(
            game_path,
            request_infos,
            max_episode_steps=max_steps,
        )
        self._env = gym.make(env_id)
        logger.info("TextWorld environment registered: %s", env_id)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        obs, infos = self._env.reset()
        self._current_step = 0
        return obs, infos

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, done, infos = self._env.step(action)
        self._current_step += 1
        return obs, float(reward), done, infos

    def get_admissible_commands(self, infos: Dict[str, Any]) -> List[str]:
        return infos.get("admissible_commands", [])

    def close(self) -> None:
        self._env.close()

    @property
    def current_step(self) -> int:
        return self._current_step
