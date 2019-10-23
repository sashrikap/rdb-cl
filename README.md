## Reward Debugging Framework

Can we formally study reward hacking behaviors, such that our self-driving cars, presumably optimizing some reward, will work nicely and safely?

README updated (19/10/17)

## Installation

```bash
pip install -e .
# To develop the code, do
pip install pre-commit pytest
pre-commit install
```

## Example Code
```bash
# Simple merge lane example
python examples/run_highway.py

# Plot reward surface over possible tasks (takes 5~6mins to run)
python examples/plot_highway.py
```

### Credit
Code reimplemented based on Prof. Dorsa Sadigh's [driving simulation](https://github.com/dsadigh/driving-interactions)
