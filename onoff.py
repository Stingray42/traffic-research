import os

os.environ['LOGURU_LEVEL'] = 'INFO'

import hurst
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
import addcopyfighandler

import model
from generator import RandomSequenceGenerator
from model import OnOffSource, autocorrelation, ExponentialCorr

if __name__ == '__main__':
    plt.rc('figure', figsize=(8, 5))
    model.dt = 1e-3
    weibull = scipy.stats.weibull_min(0.5, loc=0, scale=1.0)
    corr = ExponentialCorr(25, 1)

    generator1 = RandomSequenceGenerator(5000, weibull, corr)
    generator2 = RandomSequenceGenerator(5000, weibull, corr)

    packet_size = 1500
    data_rate = np.array([1, 2, 5, 10, 100])
    plot_colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
    plot_linestyles = ['solid', (0, (1, 1)), 'dashed', 'dashdot', (0, (10, 1))]
    plot_hatches = ['', '///', 'oo', '\\\\', '..']

    fig1, ax1 = plt.subplots(len(data_rate), 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)

    iat_data = []
    labels = []
    acorr_data = []
    hursts = []
    hursts2 = []
    rescaled_data = []
    rng = np.random.default_rng(0)
    for i in range(len(data_rate)):
        g1 = iter(generator1.start())
        g2 = iter(generator2.start())

        source = OnOffSource(
            '127.0.0.1',
            data_rate[i] * 1e6,  # mbps
            lambda: 1500,
            lambda x=g1: next(x),
            lambda x=g2: next(x),
            None
        )

        df = source.start().to_dataframe()
        iat = df['timestamp'].diff()[1:]
        iat_data.append(iat)

        label = f'${data_rate[i]} × 10^6$ МБ/с'
        labels.append(label)

        acorr = autocorrelation(iat)
        acorr_data.append(acorr)
        # acorr_range = np.arange(0, len(acorr))
        # ax3.scatter(acorr_range, acorr, color=plot_colors[i], marker='.')
        # ax3.vlines(acorr_range, 0, acorr, color=plot_colors[i])

        df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='ms')
        series = df.resample("1ms", on='timestamp').count()['bytes']
        H, c, _ = hurst.compute_Hc(series)
        hursts.append(H)
        hursts2.append(hurst.compute_Hc(iat)[0])

        rescaled = df.resample("100ms", on='timestamp').count()['bytes']
        rescaled_data.append(rescaled)

    for i, rescaled in enumerate(rescaled_data):
        rescaled.plot(ax=ax1[i], label=labels[i], color=plot_colors[i], linestyle=plot_linestyles[-i], xlabel='')

    _, bins, patches = ax2.hist(iat_data, bins=150, density=True, label=labels, edgecolor='black',
                                color=plot_colors[:len(iat_data)])
    for patch, hatch in zip(patches, plot_hatches):
        plt.setp(patch, hatch=hatch)

    ax2.set_xticks(bins)
    ax2.set_xlim(0, 15)

    for i, acorr in enumerate(acorr_data):
        ax3.plot(acorr, label=labels[i], linewidth=2.5, color=plot_colors[i], linestyle=plot_linestyles[-i])
    ax3.set_xlim(0, 100)
    ax3.scatter(np.arange(0, 100), generator1.correlation(np.arange(0, 100)), color='black', marker='o', zorder=-1,
                label='Экспоненциальная АКФ')

    ax2.legend(loc='best')
    ax3.legend(loc='best')

    for fig in fig1, fig2, fig3:
        fig.tight_layout()

    fig1.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    fig1.subplots_adjust(right=0.8)

    stat_data = {}
    for i in range(len(data_rate)):
        stat_data[data_rate[i]] = {
            'mean': iat_data[i].mean(),
            'stddev': iat_data[i].std(),
            'hurst': hursts[i],
            'hurst2': hursts2[i],
        }
    stat_data['weibull'] = {'mean': weibull.mean(), 'stddev': weibull.std()}
    print(pd.DataFrame(stat_data))

    plt.show()
