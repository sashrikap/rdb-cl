"""Distributed evaluator class. Used for parallel evaluation.
"""
import ray
import math
from rdb.infer.particles import Particles
from tqdm.auto import tqdm


class ParticleWorkerSingle(object):
    def __init__(self, env_fn, controller_fn):
        self._env_fn = env_fn
        self._env = env_fn()
        self._controller, self._runner = controller_fn(self._env)
        self._compute_result = None
        self._initialized = False

    def initialize(self):
        self._env.reset()
        temp_w = dict()
        for key in self._env.features_keys:
            temp_w[key] = 1.0
        self._controller(self._env.state, weights=temp_w)
        self._initialized = True

    def initialize_done(self):
        return self._initialized

    def compute(self, rng_key, weights, task, task_name):
        particles = Particles(
            rng_key, self._env_fn, self._controller, self._runner, weights
        )
        particles.get_features(task, task_name)
        self._compute_result = particles.dump_task(task, task_name)
        return self._compute_result

    def get_result(self):
        return self._compute_result


ParticleWorker = ray.remote(ParticleWorkerSingle)


class ParticleServer(object):
    """

    Args:
        initialize_wait (bool): wait for all workers to finish initialization

    """

    def __init__(
        self, env_fn, controller_fn, num_workers=1, parallel=True, initialize_wait=False
    ):
        self._num_workers = num_workers
        self._parallel = parallel
        if parallel:
            ray.init()
        # Define workers
        worker_cls = ParticleWorker.remote if parallel else ParticleWorkerSingle
        self._workers = [worker_cls(env_fn, controller_fn) for _ in range(num_workers)]
        self._initialize_wait = initialize_wait
        self.initialize()

    def initialize(self):
        if self._parallel:
            for worker in self._workers:
                worker.initialize.remote()
            if self._initialize_wait:
                for worker in self._workers:
                    ray.get(worker.initialize_done.remote())
        else:
            for worker in self._workers:
                worker.initialize()

    def compute_tasks(self, particles, tasks, task_names, verbose=True):
        # Filter existing tasks
        new_tasks, new_task_names = [], []
        for task, task_name in zip(tasks, task_names):
            if task_name not in particles.cached_tasks:
                new_tasks.append(task)
                new_task_names.append(task_name)
        tasks, task_names = new_tasks, new_task_names

        num_tasks = len(tasks)
        iterations = math.ceil(num_tasks / self._num_workers)

        if verbose:
            pbar = tqdm(total=num_tasks, desc="Particle Server")

        # Loop through tasks using workers
        for itr in range(iterations):
            idx_start = itr * self._num_workers
            idx_end = min((itr + 1) * self._num_workers, num_tasks)
            itr_tasks = tasks[idx_start:idx_end]
            itr_names = task_names[idx_start:idx_end]

            if self._parallel:
                # Schedule
                for wi in range(idx_end - idx_start):
                    self._workers[wi].compute.remote(
                        particles.rng_key,
                        particles.weights,
                        itr_tasks[wi],
                        itr_names[wi],
                    )
                # Retrieve
                for wi in range(idx_end - idx_start):
                    result = ray.get(self._workers[wi].get_result.remote())
                    particles.merge(result)
                    if verbose:
                        pbar.update(1)

            else:
                for wi in range(idx_end - idx_start):
                    result = self._workers[wi].compute(
                        particles.rng_key,
                        particles.weights,
                        itr_tasks[wi],
                        itr_names[wi],
                    )
                    particles.merge(result)
                    if verbose:
                        pbar.update(1)

        if verbose:
            pbar.close()

        return particles
