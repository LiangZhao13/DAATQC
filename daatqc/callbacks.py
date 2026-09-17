"""Training callbacks used by both initial and continued training."""

from __future__ import annotations

import csv
import os
import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggerCallback(BaseCallback):
    def __init__(
        self,
        print_every_episodes: int = 10,
        csv_path: str = "training_log.csv",
        rolling_window: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.print_every_episodes = print_every_episodes
        self.csv_path = csv_path
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=rolling_window)
        self.episode_lengths = deque(maxlen=rolling_window)
        self.interval_successes = 0
        self.interval_episodes = 0
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        directory = os.path.dirname(self.csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    [
                        "episodes", "total_timesteps", "time_elapsed_sec", "fps",
                        "ep_len_mean", "ep_rew_mean", "success_count", "success_rate",
                        "actor_loss", "critic_loss", "ent_coef", "ent_coef_loss",
                        "learning_rate", "n_updates",
                    ]
                )

    def _metric(self, key: str):
        value = self.model.logger.name_to_value.get(key, np.nan)
        return np.nan if value is None else value

    def _write_status(self) -> None:
        elapsed = time.time() - self.start_time
        frames_per_second = self.num_timesteps / max(elapsed, 1e-8)
        length_mean = float(np.mean(self.episode_lengths)) if self.episode_lengths else np.nan
        reward_mean = float(np.mean(self.episode_rewards)) if self.episode_rewards else np.nan
        success_rate = self.interval_successes / max(self.interval_episodes, 1)
        metrics = [
            self._metric("train/actor_loss"),
            self._metric("train/critic_loss"),
            self._metric("train/ent_coef"),
            self._metric("train/ent_coef_loss"),
            self._metric("train/learning_rate"),
            self._metric("train/n_updates"),
        ]
        row = [
            self.episode_count,
            int(self.num_timesteps),
            round(elapsed, 2),
            round(frames_per_second, 2),
            length_mean,
            reward_mean,
            self.interval_successes,
            success_rate,
            *metrics,
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)
        print(
            f"episodes={self.episode_count:4d} | timesteps={self.num_timesteps:8d} | "
            f"fps={frames_per_second:6.1f} | ep_len_mean={length_mean:7.2f} | "
            f"ep_rew_mean={reward_mean:11.3f} | success_rate={100.0 * success_rate:5.1f}%"
        )
        self.interval_successes = 0
        self.interval_episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []) or []:
            if isinstance(info, dict) and "episode" in info:
                self.episode_count += 1
                self.interval_episodes += 1
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.interval_successes += int(bool(info.get("is_success", False)))
                if self.episode_count % self.print_every_episodes == 0:
                    self._write_status()
        return True

