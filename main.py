import os.path

os.environ['LOGURU_LEVEL'] = 'INFO'

import model

from datetime import datetime

import dill
import hurst
import scipy.stats

from model import ExponentialCorr

from loguru import logger

from plots import *


def random_ipv4(rng: np.random.Generator = np.random.default_rng(0)) -> str:
    return ".".join(str(rng.integers(0, 255)) for _ in range(4))


if __name__ == '__main__':
    filename = 'multiplexer.bin'
    if os.path.exists(filename):
        logger.info('load computed data from file: {}', filename)
        start = datetime.now()
        with open(filename, 'rb') as file:
            multiplexer = dill.load(file)
        end = datetime.now()
        logger.info('loading complete in {}', end - start)
    else:
        model.dt = 1e-3
        weibull = scipy.stats.weibull_min(0.5, loc=0, scale=1.0)
        sources = []
        corr = ExponentialCorr(25, 1)
        rng = np.random.default_rng(0)
        for i in range(2):
            generator1 = RandomSequenceGenerator(5000, weibull, corr)
            generator2 = RandomSequenceGenerator(5000, weibull, corr)
            g1 = iter(generator1.start())
            g2 = iter(generator2.start())
            logger.debug('on generator: {}, off generator: {}', g1, g2)
            source = OnOffSource(
                random_ipv4(rng),
                1_000_000,
                lambda: 1500,
                lambda x=g1: next(x),
                lambda x=g2: next(x),
                None
            )
            sources.append(source)
            # plot_on_off_source(source, generator1, weibull_pdf)
            # plot_source_stats(source)

            # df = source.start().to_dataframe()
            # df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='ms')

        multiplexer = Multiplexer(sources, 1_000_000)
        df = multiplexer.start().to_dataframe()
        df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='ms')
        series = df.resample("1ms", on='timestamp').count()['bytes']
        H, c, _ = hurst.compute_Hc(series)
        print(H)
        # df.to_csv('data.csv', index=False)
        # with open(filename, 'wb') as file:
        #     dill.dump(multiplexer, file)

    plot_multiplexed_packets(multiplexer)
    plot_multiplexed_stats(multiplexer)
