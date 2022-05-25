from dataclasses import dataclass
from typing import ClassVar, List

import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import fsolve

from model import Correlation, autocorrelation
from scipy.stats._distn_infrastructure import rv_frozen

rv_frozen.__str__ = lambda self: f'{self.dist.name}{self.args}{self.kwds}'


@dataclass
class RandomSequenceGenerator:
    L: int
    distribution: rv_frozen
    correlation: Correlation
    _seed: int = None
    _result: NDArray[float] = None

    _used_seeds: ClassVar[List[int]] = []

    def set_seed(self, seed: int):
        assert not self._seed, f'seed was already set to {self._seed}'
        self._seed = seed

    def start(self) -> NDArray[float]:
        if self._result is not None:
            return self._result

        n = 201
        x = np.linspace(-4, 4, n)
        y = np.zeros(n)

        for k in range(n):
            y[k] = fsolve(lambda t: self.distribution.cdf(t) - scipy.stats.norm.cdf(x[k]), 0)

        poly = np.polyfit(x, y, deg=10)
        transform = np.vectorize(lambda t: np.polyval(poly, t))

        if not self._seed:
            try:
                seeds = pd.read_csv('seeds.csv')
                matched_seeds = seeds[
                    (seeds['L'] == self.L)
                    & (seeds['tau0'] == self.correlation.tau0)
                    & (seeds['sigma'] == self.correlation.sigma)
                    & (seeds['corr'] == self.correlation.name)
                    & (seeds['cdf'] == str(self.distribution))
                    ]['seed']
            except Exception as e:
                logger.error('{}: {}', type(e).__name__, e)
                seeds = pd.DataFrame()
                matched_seeds = []

            seed = None
            found = False
            for seed in matched_seeds:
                seed = int(seed)
                if seed in RandomSequenceGenerator._used_seeds:
                    logger.debug('seed {} was already used', seed)
                else:
                    logger.info('pre-computed seed: {}', seed)
                    found = True
                    break

            if not found:
                logger.info('start seed search')
                seed = 0
                p = 0
                mse = np.inf
                while p <= 0.05 or mse > 0.001:
                    noise = np.random.default_rng(seed).standard_normal(self.L)
                    sequence = transform(self.correlation.generate(noise))
                    _, p = scipy.stats.ks_1samp(sequence, self.distribution.cdf)
                    exper_corr = autocorrelation(sequence)[0:self.correlation.tau0 * 2]
                    theor_corr = self.correlation(np.arange(0, len(exper_corr)))
                    mse = ((exper_corr - theor_corr) ** 2).mean()
                    logger.debug('seed: {},\tp: {},\tmse: {}', seed, p, mse)
                    seed = seed + 1
                    if seed in RandomSequenceGenerator._used_seeds:
                        seed += 1

                logger.info('seed: {},\tp: {},\tmse: {}', seed, p, mse)
                seeds = seeds.append({
                    'L': self.L,
                    'tau0': self.correlation.tau0,
                    'sigma': self.correlation.sigma,
                    'corr': self.correlation.name,
                    'cdf': str(self.distribution),
                    'seed': seed
                }, ignore_index=True)
                seeds.to_csv('seeds.csv', index=False)

            RandomSequenceGenerator._used_seeds.append(seed)
            self._seed = seed

        noise = np.random.default_rng(self._seed).standard_normal(self.L)
        correlated_noise = self.correlation.generate(noise)
        result = transform(correlated_noise)
        if (result < 0).any():
            logger.warning('result contains negative values: {}', result[result < 0])
        self._result = np.clip(result, 0.0, None)
        return self._result

    def __iter__(self):
        return iter(self.start())
