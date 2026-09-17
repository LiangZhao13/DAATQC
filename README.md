# DAATQC

**Demand-Actuator Attention Truncated Quantile Critics (DAATQC)** is an
end-to-end deep reinforcement learning controller for three-degree-of-freedom
dynamic positioning with six heterogeneous thrusters. The actor directly
outputs normalized rotational-speed commands, so no separate online thrust
allocation solver is required.

This repository is the modular Python implementation accompanying the DAATQC
study. It targets **Python 3.10** and separates simulation, representation
learning, initial training, continuation training, and evaluation.

## Repository structure

```text
DAATQC/
├── daatqc/
│   ├── attention.py       # Demand-Actuator Attention feature extractor
│   ├── callbacks.py       # CSV episode logger
│   ├── config.py          # Reproducible dataclass configurations
│   ├── disturbances.py    # Current, stochastic wind, and JONSWAP wave drift
│   ├── environment.py     # 14-D Gymnasium DP environment and reward
│   ├── thrusters.py       # Six heterogeneous thrusters and actuator lag
│   ├── training.py        # Shared TQC, environment, and callback factories
│   ├── utils.py           # Angle and rotation helpers
│   └── vessel.py          # 3-DOF low-speed supply-vessel model
├── train.py               # Initial 1.2-million-step training entry point
├── continue_train.py      # 0.5-million-step strict-accuracy continuation
├── evaluate.py            # Deterministic 300 s four-corner test
├── requirements.txt
└── tests/
```

## Observation and action spaces

The policy observation remains exactly 14-dimensional:

1. body-frame position and heading errors (3);
2. surge, sway, and yaw velocities (3);
3. six actual thruster speeds normalized by their individual limits (6); and
4. two reserved environmental channels (2).

The last two positions preserve compatibility with the original DAATQC input,
but are zero by default. Current, wind, and wave values are therefore **not fed
directly to the policy**. Current changes the water-relative vessel velocity,
whereas wind and waves enter as generalized forces. The policy observes only
their consequences through vessel pose and velocity. Setting
`EnvironmentConfig(expose_current_to_policy=True)` explicitly restores the
legacy current channels without changing the 14-dimensional shape. The action
is a six-dimensional vector in `[-1, 1]`, mapped to each thruster's physical
rotational-speed limit.

## Environmental disturbances

The disturbance module was integrated from the `TQC_energy_env` environment:

- ocean current modifies the water-relative vessel velocity;
- wind uses relative wind velocity, direction-dependent aerodynamic
  coefficients, and a first-order Gauss-Markov speed process; and
- irregular waves are generated from a discretized JONSWAP spectrum and
  converted into low-frequency drift forces.

For reproducibility of the reported training setup, the defaults are:

```text
current speed = 0 m/s
mean wind speed = 0 m/s
significant wave height = 0 m
```

Non-zero values can be supplied from the command line for robustness studies,
without changing the policy architecture or observation dimension.

## Installation

Create a Python 3.10 environment. On Linux:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For an NVIDIA A10, install the PyTorch build matching the machine's CUDA driver
before installing the remaining requirements if the default PyPI wheel is not
appropriate. Verify GPU availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Initial training

The default command reproduces the notebook settings: four vectorized
environments, 1,200 steps per episode, and 1.2 million interactions.

```bash
python train.py
```

Useful overrides include:

```bash
python train.py --timesteps 1200000 --n-envs 4 --device cuda
```

The default output directory is `drl_dp_DAATQC_results/`. It contains the final
model, best evaluation model, checkpoints, evaluation logs, and
`training_log.csv`.

## Continuation training

Continuation training tightens the success criteria to 0.20 m, 2 degrees, a
speed threshold of 0.08, and 50 consecutive successful steps:

```bash
python continue_train.py
```

The script automatically loads `DAATQC_final.zip` and falls back to
`best_model/best_model.zip`. An explicit checkpoint can also be supplied:

```bash
python continue_train.py \
  --model-path ./drl_dp_DAATQC_results/best_model/best_model.zip \
  --timesteps 500000
```

The continued model is saved as `DAATQC_continued_final.zip`.

## Four-corner evaluation

```bash
python evaluate.py \
  --model-path ./drl_dp_DAATQC_results/DAATQC_continued_final.zip
```

The evaluation holds each reference for 50 s and saves a CSV time history and
an SVG trajectory. To test unobserved wind and wave disturbances, for example:

```bash
python evaluate.py \
  --model-path ./drl_dp_DAATQC_results/DAATQC_continued_final.zip \
  --current-speed 0.5 --wind-speed 3.0 --wave-height 0.3
```

## Tests

```bash
pytest -q
```

The tests confirm that the observation remains 14-dimensional, the default
training disturbances are zero, and non-zero current, wind, and waves affect
the vessel dynamics without being directly appended to the policy input.
