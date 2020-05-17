"""Distributed designer class, used for parallel sampling of designer.
"""


import ray
import math
import numpy as onp
import jax.numpy as np
from time import time
from jax import random


class DesignerWorkerSingle(object):
    def __init__(self, designer_fn):
        self._designer = designer_fn()
        self._samples = None
        self._key = None
        self._true_w = None
        self._rng_name = None
        self._prior_tasks = None

    def simulate(self, tasks, save_name, itr, tqdm_position=0):
        samples = self._designer.simulate(
            tasks, save_name=save_name, tqdm_position=tqdm_position
        )
        self._samples = samples.weights

    def get_samples(self):
        return self._samples

    def update_key(self, rng_key):
        self._designer.update_key(rng_key)
        self._key = rng_key

    def get_key(self):
        return self._key

    def set_true_w(self, true_w):
        self._designer.true_w = true_w
        self._true_w = true_w

    def get_true_w(self):
        return self._true_w

    def set_prior_tasks(self, tasks):
        self._designer.prior_tasks = tasks
        self._prior_tasks = tasks

    def get_prior_tasks(self):
        return self._prior_tasks

    def set_rng_name(self, rng_name):
        self._designer.rng_name = rng_name
        self._rng_name = rng_name

    def get_rng_name(self):
        return self._rng_name


DesignerWorker = ray.remote(DesignerWorkerSingle)


class DesignerServer(object):
    """Simulate multiple designers in parallel.

    Require:
        # Server = # active functions

    """

    def __init__(self, designer_fn, parallel=True):
        self._parallel = parallel
        self._designer_fn = designer_fn
        self._local_designer = designer_fn()
        self._worker_cls = DesignerWorker.remote if parallel else DesignerWorkerSingle
        self._workers = []
        if self._parallel:
            try:
                ray.init()
            except:
                pass

    def register(self, num_workers):
        if not self._parallel:
            num_workers = 1
        self._workers = [
            self._worker_cls(self._designer_fn) for _ in range(num_workers)
        ]

    def simulate(self, tasks, methods, itr):
        """Simulate designer on tasks.

        Args:
            tasks (ndarray):
                parallel = True - (n_designers, n_tasks, task_dim)
                parallel = False - (n_tasks, task_dim)

        """
        tasks = np.array(tasks)
        assert len(tasks.shape) == 3
        if self._parallel:
            assert len(tasks) == len(self._workers)
            assert len(tasks) == len(methods)
        else:
            assert len(tasks) == 2

        t_start = time()

        samples = []
        save_names = {}
        if self._parallel:
            for wi in range(len(self._workers)):
                save_names[wi] = f"designer_method_{methods[wi]}_itr_{itr:02d}"
                self._workers[wi].simulate.remote(
                    tasks[wi], save_name=save_names[wi], itr=itr, tqdm_position=wi
                )

            for wi in range(len(self._workers)):
                sample_ws = ray.get(self._workers[wi].get_samples.remote())
                particles = self._local_designer.create_particles(
                    sample_ws,
                    controller=self._local_designer._one_controller,
                    runner=self._local_designer._one_runner,
                    save_name=save_names[wi],
                )
                samples.append(particles)
        else:
            for mi in range(len(methods)):
                # Each method
                samples.append(
                    self.workers[0].simulate(
                        onp.array(tasks[mi]),
                        save_name=f"designer_method_{methods[mi]}_itr_{itr:02d}",
                    )
                )

        t_start = time()
        print(f"Simulating {len(tasks)} designers finished {(time() - t_start):.3f}s")

        return samples

    @property
    def designer(self):
        return self._local_designer

    def update_key(self, rng_key):
        if self._parallel:
            n_workers = len(self._workers)
            all_keys = random.split(rng_key, n_workers + 1)
            self._local_designer.update_key(all_keys[-1])

            for wi in range(n_workers):
                self._workers[wi].update_key.remote(all_keys[wi])

            for wi in range(n_workers):
                ray.get(self._workers[wi].get_key.remote())

        else:
            all_keys = random.split(rng_key, 2)
            self._local_designer.update_key(all_keys[0])
            if len(self._workers) > 0:
                self.workers[0].update_key(all_keys[1])

    def set_true_w(self, true_w):
        self._local_designer.true_w = true_w
        if self._parallel:
            for wi in range(len(self._workers)):
                self._workers[wi].set_true_w.remote(true_w)

            for wi in range(len(self._workers)):
                ray.get(self._workers[wi].get_true_w.remote())

        else:
            self.workers[0].set_true_w(true_w)

    def set_prior_tasks(self, tasks):
        self._local_designer.prior_tasks = tasks
        if self._parallel:
            for wi in range(len(self._workers)):
                self._workers[wi].set_prior_tasks.remote(tasks)

            for wi in range(len(self._workers)):
                ray.get(self._workers[wi].get_prior_tasks.remote())

        else:
            self.workers[0].set_prior_tasks(tasks)

    def set_rng_name(self, rng_name):
        if self._parallel:
            for wi in range(len(self._workers)):
                self._workers[wi].set_rng_name.remote(rng_name)

            for wi in range(len(self._workers)):
                ray.get(self._workers[wi].get_rng_name.remote())

        else:
            self.workers[0].set_rng_name(rng_name)
