## Reward Debugging Framework

Can we formally study reward hacking behaviors, such that our self-driving cars, presumably optimizing some reward, will work nicely and safely?

README updated (19/02/04)

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
python examples/run_simple.py
python examples/run_optimal_control.py
```


### Citing

If you find this codebase useful in your research, please consider citing:

	@article{he2021assisted,
  	    title={Assisted Robust Reward Design},
  	    author={He, Jerry Zhi-Yang He and Dragan D. Anca},
  	    journal={arXiv preprint arXiv:2111.09884},
  	    year={2021}
	}


### Credit
Code reimplemented based on Prof. Dorsa Sadigh's [driving simulation](https://github.com/dsadigh/driving-interactions)
