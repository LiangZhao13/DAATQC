"""Demand-Actuator Attention feature extractor used by DAATQC."""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DemandActuatorAttentionExtractor(BaseFeaturesExtractor):
    """Couple three control-demand tokens with six heterogeneous actuators."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        if observation_space.shape != (14,):
            raise ValueError("DemandActuatorAttentionExtractor requires a 14-D observation")

        self.embedding_dim = 64
        self.number_of_heads = 4
        self.saturation_start = 0.75
        self.saturation_value_strength = 0.80

        vessel_length = 76.2
        main_indicator = np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)
        energy_weight = np.array([1, 1, 1, 1, 10, 10], dtype=np.float32)
        surge_contribution = np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)
        sway_contribution = np.array([1, 1, 1, 1, 0, 0], dtype=np.float32)
        yaw_arm = np.array([30, 22, -22, -30, -8, 8], dtype=np.float32)
        thrust_coefficient = np.array([3.2, 3.2, 3.2, 3.2, 31.2, 31.2], dtype=np.float32)
        actuator_static = np.stack(
            [
                main_indicator,
                energy_weight / energy_weight.max(),
                surge_contribution,
                sway_contribution,
                yaw_arm / (vessel_length / 2.0),
                thrust_coefficient / thrust_coefficient.max(),
            ],
            axis=1,
        )
        self.register_buffer(
            "actuator_static_features", th.tensor(actuator_static, dtype=th.float32)
        )
        self.register_buffer("energy_weight", th.tensor(energy_weight, dtype=th.float32))

        self.demand_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(),
        )
        self.actuator_encoder = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(),
        )
        self.demand_identity_embedding = nn.Parameter(th.zeros(1, 3, 64))
        self.actuator_identity_embedding = nn.Parameter(th.zeros(1, 6, 64))
        nn.init.normal_(self.demand_identity_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.actuator_identity_embedding, mean=0.0, std=0.02)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=self.number_of_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(64)
        self.feedforward = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        self.feedforward_norm = nn.LayerNorm(64)
        self.fusion = nn.Sequential(
            nn.Linear(258, features_dim), nn.LayerNorm(features_dim), nn.SiLU()
        )

    @staticmethod
    def _demand_tokens(observation: th.Tensor) -> th.Tensor:
        surge = th.stack(
            [observation[:, 0] / 10.0, observation[:, 3] / 2.0, observation[:, 12]],
            dim=1,
        )
        sway = th.stack(
            [observation[:, 1] / 10.0, observation[:, 4] / 2.0, observation[:, 13]],
            dim=1,
        )
        yaw = th.stack(
            [
                observation[:, 2] / np.pi,
                observation[:, 5] / 0.20,
                th.zeros_like(observation[:, 0]),
            ],
            dim=1,
        )
        return th.clamp(th.stack([surge, sway, yaw], dim=1), -5.0, 5.0)

    def _actuator_tokens(self, observation: th.Tensor):
        signed_speed = th.clamp(observation[:, 6:12], -1.5, 1.5)
        speed_magnitude = th.abs(signed_speed)
        remaining_margin = th.clamp(1.0 - speed_magnitude, 0.0, 1.0)
        saturation = th.clamp(
            (speed_magnitude - self.saturation_start) / (1.0 - self.saturation_start),
            0.0,
            1.0,
        )
        static = self.actuator_static_features.unsqueeze(0).expand(
            observation.shape[0], -1, -1
        )
        dynamic = th.stack(
            [signed_speed, speed_magnitude, remaining_margin, saturation], dim=2
        )
        return th.cat([dynamic, static], dim=2), saturation, remaining_margin, speed_magnitude

    def forward(self, observation: th.Tensor) -> th.Tensor:
        demand_input = self._demand_tokens(observation)
        actuator_input, saturation, margin, speed = self._actuator_tokens(observation)
        demand = self.demand_encoder(demand_input) + self.demand_identity_embedding
        actuator = self.actuator_encoder(actuator_input) + self.actuator_identity_embedding

        availability = 1.0 - self.saturation_value_strength * saturation
        attention, _ = self.cross_attention(
            query=demand,
            key=actuator,
            value=actuator * availability.unsqueeze(-1),
            need_weights=False,
        )
        contextual_demand = self.attention_norm(demand + attention)
        contextual_demand = self.feedforward_norm(
            contextual_demand + self.feedforward(contextual_demand)
        )

        pooling_weight = 0.5 + 0.5 * margin
        actuator_summary = (
            actuator * pooling_weight.unsqueeze(-1)
        ).sum(dim=1) / (pooling_weight.sum(dim=1, keepdim=True) + 1e-8)
        global_saturation = 0.5 * saturation.mean(dim=1, keepdim=True) + 0.5 * (
            saturation.max(dim=1, keepdim=True).values
        )
        energy_proxy = (self.energy_weight * speed.pow(3)).sum(
            dim=1, keepdim=True
        ) / self.energy_weight.sum()
        fused = th.cat(
            [
                contextual_demand.reshape(observation.shape[0], -1),
                actuator_summary,
                global_saturation,
                energy_proxy,
            ],
            dim=1,
        )
        return self.fusion(fused)

