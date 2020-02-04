from rdb.infer.algos import *


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Methods:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel, proposal, sample_method="mh", sample_args={}):
        self._rng_key = rng_key
        self._sampler = self._build_sampler(
            kernel, proposal, sample_method, sample_args
        )

    def update_key(self, rng_key):
        self._rng_key = rng_key
        # self._sampler.update_key(rng_key)

    def _build_sampler(self, kernel, proposal, sample_method, sample_args):
        if sample_method == "mh":
            return MetropolisHasting(
                self._rng_key, kernel, proposal=proposal, **sample_args
            )
        else:
            raise NotImplementedError
