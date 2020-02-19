"""Distributed evaluator class. Used for parallel evaluation.

Useful for:
    * Evaluate each belief point on high # of tasks

General workflow:
    * Experiment -> ParticleServer -> ParticleWorker
      Experiment <- ParticleServer <-

"""
import ray
import math
from rdb.infer import *
from tqdm.auto import tqdm


class ParticleWorkerSingle(object):
    def __init__(self, env_fn, controller_fn, normalized_key, weight_params, max_batch):
        self._env_fn = env_fn
        self._env = env_fn()
        self._controller, self._runner = controller_fn(self._env)
        self._compute_result = None
        self._initialized = False
        self._max_batch = max_batch

    def initialize(self):
        self._env.reset()
        temp_w = dict()
        for key in self._env.features_keys:
            temp_w[key] = [1.0]
        init_states = onp.repeat(self._env.state, self._max_batch, axis=0)
        weights_arr = DictList(temp_w).repeat(self._max_batch, axis=0).numpy_array()
        actions = self._controller(init_states, weights=None, weights_arr=weights_arr)
        self._runner(init_states, actions, weights=None, weights_arr=weights_arr)
        self._initialized = True

    def initialize_done(self):
        return self._initialized

    def compute(self, weights_arr, states):
        """Compute weights * tasks.

        Note:
            * In general good to avoid re-compiling.

        """
        actions, costs, feats, feats_sum, violations = collect_trajs(
            weights_arr,
            states,
            self._controller,
            self._runner,
            max_batch=self._max_batch,
        )
        self._compute_result = dict(
            actions=actions,
            costs=costs,
            feats=feats,
            feats_sum=feats_sum,
            violations=violations,
        )

    def get_result(self):
        return self._compute_result


ParticleWorker = ray.remote(ParticleWorkerSingle)


def merge_result(result, new_result):
    assert new_result is not None
    if result is None:
        return new_result
    else:
        for key, val in result.items():
            if isinstance(val, DictList):
                result[key] = val.concat(new_result[key], axis=0)
            else:
                result[key] = np.concatenate([val, new_result[key]], axis=0)
        return result


class ParticleServer(object):
    """

    Args:
        initialize_wait (bool): wait for all workers to finish initialization

    """

    def __init__(
        self,
        env_fn,
        controller_fn,
        num_workers=1,
        parallel=True,
        initialize_wait=False,
        max_batch=1000,
        normalized_key=None,
        weight_params={},
    ):
        self._num_workers = num_workers
        self._parallel = parallel
        self._max_batch = max_batch
        self._env = env_fn()
        if parallel:
            ray.init()
        else:
            # Force num_workers = 1
            self.num_workers = 1
        # Define workers
        worker_cls = ParticleWorker.remote if parallel else ParticleWorkerSingle
        self._workers = [
            worker_cls(env_fn, controller_fn, normalized_key, weight_params, max_batch)
            for _ in range(num_workers)
        ]
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

    def compute_tasks(self, particles, tasks, verbose=True):
        """Compute all tasks for all particle weights.

        Note:
            * Run forward passes nweights * ntasks / nbatch times.
            * Has nworkers.

        """
        if particles is None:
            return
        # Filter existing tasks
        tasks = onp.array(tasks)
        new_tasks = []
        for task in tasks:
            if particles.get_task_name(task) not in particles.cached_names:
                new_tasks.append(task)
        tasks = onp.array(new_tasks)

        # Batch nweights * ntasks
        states = self._env.get_init_states(onp.array(tasks))
        assert len(tasks.shape) == 2
        ntasks, nweights = len(tasks), len(particles.weights)
        nfeats = len(self._env.features_keys)
        task_dim = tasks.shape[1]
        state_dim = states.shape[1]
        #  batch_states (ntasks * nweights, state_dim)
        #  batch_weights nfeats * (ntasks * nweights)
        batch_states, batch_weights = cross_product(
            states, particles.weights, onp.array, DictList
        )
        batch_weights = batch_weights.prepare(self._env.features_keys)

        result = None
        if self._parallel:
            # Schedule
            num_batch_tasks = len(batch_states)
            tasks_per_worker = math.ceil(num_batch_tasks / self._num_workers)

            # Loop through workers
            for wi in range(len(self._workers)):
                idx_start = wi * tasks_per_worker
                idx_end = (wi + 1) * tasks_per_worker
                states = batch_states[idx_start:idx_end]
                weights = batch_weights[idx_start:idx_end]
                weights_arr = weights.numpy_array()
                assert states.shape == (tasks_per_worker, state_dim)
                assert weights_arr.shape == (nfeats, tasks_per_worker)
                self._workers[wi].compute.remote(weights_arr, states)
            # Retrieve
            for wi in range(len(self._workers)):
                result_w = ray.get(self._workers[wi].get_result.remote())
                result = merge_result(result, result_w)
        else:
            batch_weights_arr = batch_weights.numpy_array()
            result_w = self._workers[0].compute(batch_weights_arr, batch_states)
            result = merge_result(result, result_w)

        #  result["actions"] -> (ntasks, nweights, T, udim)
        for key, val in result.items():
            val_shape = val.shape
            #   shape (ntasks, nweights, ...)
            result[key] = val.reshape((ntasks, nweights) + val_shape[1:])
        particles.merge_bulk_tasks(tasks, result)
        return particles
