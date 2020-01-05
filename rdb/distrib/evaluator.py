"""Distributed evaluator class. Used for parallel evaluation.
"""
import ray
from rdb.infer.particles import Particles


class Evaluator(object):
    def __init__(self):
        self._eval_result = None

    def evaluate(self):
        pass


@ray.remote
class EvaluatorParticle(Evaluator):
    def __init__(self, env_fn, controller_fn):
        super().__init__()
        self._env = env_fn()
        self._controller, self._runner = controller_fn(self._env)

    def evaluate(self, rng_key, sample_ws, target_w, task, task_name):
        particles = Particles(
            rng_key, self._env, self._controller, self._runner, sample_ws
        )
        self._eval_result = particles.compare_with(task, task_name, target_w)

    def get_result(self):
        return self._eval_result
