## Reward Hacking in Simulated Autonomous Vehicle Environments

Studying the generalizability of reward functions in simulated autonomous vehicle environments by 1) examining if transferring reward functions across environments is robust to environment ordering and beneficial to the model’s performance and 2) determining what methods serve as a proxy for better environment orderings.

## Installation

```bash
# Pre-requisite: you need to install conda
conda create python=3.9 -n [name for environment]
pip install -e .
# To develop the code, do
pip install pre-commit pytest
pre-commit install
```

## Example Code
```bash
# Simple merge lane example
python examples/run_simple.py
python examples/run_optimal_control.py
```


### Credit
Code reimplemented based on Jerry Zhi-Yang He's implementation of [assisted robust reward design](https://arxiv.org/abs/2111.09884) and Prof. Dorsa Sadigh's [driving simulation](https://github.com/dsadigh/driving-interactions)
