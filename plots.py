import matplotlib.ticker
import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import addcopyfighandler

from typing import Tuple, Callable, List, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from main import OnOffSource, autocorrelation, RandomSequenceGenerator, tx_delay, Multiplexer

_COLORS = plt.style.library['bmh']['axes.prop_cycle'].by_key()['color']
_HATCHES = ['', '///', 'o', '\\\\', '.']
sns.set_style('whitegrid')
plt.rc('axes', prop_cycle=plt.cycler(color=_COLORS))
plt.rc('font', family=['DejaVu Sans'])


# plt.rc('hatch', color='black', linewidth=2)


def _color(i: int):
    return _COLORS[i % len(_COLORS)]


def _hatch(i: int):
    return _HATCHES[i % len(_HATCHES)]


# plt.rc('figure', figsize=(8, 6), dpi=200)

def plot_on_off_source(source: OnOffSource,
                       generator1: RandomSequenceGenerator,
                       pdf: Callable[[NDArray[float]], NDArray[float]]):
    fig, ax = plt.subplots(3, 1)
    ax[0].title.set_text('On/Off source')

    ax[0].set_xlabel("Sample, $n$")
    ax[1].set_xlabel('Value, $x$')
    ax[2].set_xlabel(r'Lag, $\tau$')

    t = matplotlib.transforms.ScaledTranslation(-70 / 72, 0, fig.dpi_scale_trans)
    ax[0].text(0, 0.5, 'a)', transform=ax[0].transAxes + t)
    ax[1].text(0, 0.5, 'b)', transform=ax[1].transAxes + t)
    ax[2].text(0, 0.5, 'c)', transform=ax[2].transAxes + t)

    t = matplotlib.transforms.ScaledTranslation(-50 / 72, 0, fig.dpi_scale_trans)
    ax[0].text(0, 0.5, 'Value, $x$', transform=ax[0].transAxes + t, rotation=90, va='center')
    ax[1].text(0, 0.5, 'Count, $f(x)$', transform=ax[1].transAxes + t, rotation=90, va='center')
    ax[2].text(0, 0.5, r'Auto-correlation, $R(\tau)$', transform=ax[2].transAxes + t, rotation=90, va='center')

    source.start()
    off = np.array(source.off_durations)
    on = np.array(source.on_durations)
    _plot_random_sequence(ax[0], off, color=_color(0))
    _plot_random_sequence(ax[0], on, color=_color(3))
    _plot_pdf(ax[1], on, off, pdf)
    _plot_correlation(ax[2], on, off, generator1)

    fig.tight_layout()
    fig.legend(loc='upper left')
    plt.show()


def _plot_random_sequence(ax: Axes, sequence: NDArray[float], **kwargs):
    ax.plot(sequence, **kwargs)
    ax.set_xlim(0, len(sequence))


def _plot_pdf(ax: Axes, on: NDArray[float], off: NDArray[float], pdf: Callable[[NDArray[float]], NDArray[float]]):
    count, bins, _ = ax.hist([on, off], bins=50, color=[_color(0), _color(3)], label=['ON-период', 'OFF-период'])

    pdf_range = np.linspace((bins[0] + bins[1]) / 2, (bins[-1] + bins[-2]) / 2, 1000)
    pdf = pdf(pdf_range)
    scale = count.max() / pdf.max()
    ax.plot(pdf_range, pdf * scale, color=_color(1))
    ax.set_xlim(bins[0], bins[-1])


def _plot_correlation(ax: Axes, on: NDArray[float], off: NDArray[float], generator: Optional[RandomSequenceGenerator]):
    d_shift = 0.2
    on_corr = autocorrelation(on)
    on_corr[::2] = np.nan
    corr_range = np.arange(0, len(on_corr))
    ax.scatter(corr_range - d_shift, on_corr, color=_color(0), marker='.')
    ax.vlines(corr_range - d_shift, 0, on_corr, color=_color(0))

    off_corr = autocorrelation(off)
    off_corr[::2] = np.nan
    corr_range = np.arange(0, len(off_corr))
    ax.scatter(corr_range + d_shift, off_corr, color=_color(3), marker='.')
    ax.vlines(corr_range + d_shift, 0, off_corr, color=_color(3))

    if generator:
        theor_corr = generator.corr.correlation(corr_range)
        ax.plot(corr_range, theor_corr, color=_color(1), label='Теоретические данные')

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.25, 1)


def plot_multiplexed_packets(multiplexer: Multiplexer):
    time_limit = 100
    fig, ax = plt.subplots(len(multiplexer.sources) + 1, 1, sharex=True)
    for a in ax:
        a.set_xlim(0, time_limit)
        a.grid(False)

    multiplexed = multiplexer.start()
    for i, source in enumerate(multiplexer.sources):
        plot_config = {
            'color': _color(i),
            'hatch': _hatch(i),
            'edgecolor': 'black',
            'alpha': 0.9
        }
        data_before = []
        for p in source.start():
            if p.timestamp >= time_limit:
                break
            data_before.append((p.timestamp, tx_delay(p.bytes, source.data_rate)))
        ax[i].broken_barh(data_before, (0, 1), **plot_config)

        data_after = []
        for p in multiplexed:
            if p.timestamp >= time_limit:
                break
            if p.src_ip == source.src_ip:
                data_after.append((p.timestamp, tx_delay(p.bytes, multiplexer.data_rate)))
        ax[-1].broken_barh(data_after, (0, 1), **plot_config)

    fig.tight_layout()
    plt.show()


def _plot_on_off_sequence(ax: Axes, off_durations: List[float], on_durations: List[float], limit: int, **kwargs):
    starts = []
    widths = on_durations[0:limit]
    for i, off in enumerate(off_durations[0:limit]):
        value = off
        if i > 0:
            value += on_durations[i - 1]
        starts.append(value)

    ax.broken_barh(list(zip(starts, widths)), (0, 1), **kwargs)


def plot_multiplexed_stats(multiplexer: Multiplexer):
    fig, ax = plt.subplots(3, 1)

    packets = multiplexer.start()
    df = packets.to_dataframe()

    off = df['timestamp'].diff().fillna(0)
    _plot_random_sequence(ax[0], off)
    ax[1].hist(off, bins=np.linspace(off.min(), off.max(), 50))
    ax[1].set_ylim(0, 1000)

    on_corr = autocorrelation(off)
    on_corr[1::2] = np.nan
    corr_range = np.arange(0, len(on_corr))
    ax[2].scatter(corr_range, on_corr, color=_color(0), marker='.')
    ax[2].vlines(corr_range, 0, on_corr, color=_color(0))
    ax[2].set_xlim(-0.5, 100)
    ax[2].set_ylim(-0.25, 1.05)

    fig.tight_layout()
    plt.show()


def plot_source_stats(source: OnOffSource):
    fig, ax = plt.subplots(3, 1)

    packets = source.start()
    df = packets.to_dataframe()

    off = df['timestamp'].diff()[1:]
    _plot_random_sequence(ax[0], off)
    counts, bins, bars = ax[1].hist(off, bins=np.linspace(off.min(), off.max(), 50))
    ylim = 1000
    for i, count in enumerate(counts):
        if count > ylim:
            x = (bins[i] + bins[i + 1]) / 2
            y = 900
            a = ax[1].annotate(str(count), (x, y), ha="center")
            a.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round'))

    ax[1].set_ylim(0, ylim)

    on_corr = autocorrelation(off)
    corr_range = np.arange(0, len(on_corr))
    ax[2].scatter(corr_range, on_corr, color=_color(0), marker='.')
    ax[2].vlines(corr_range, 0, on_corr, color=_color(0))
    ax[2].set_xlim(-0.5, 50)
    ax[2].set_ylim(-0.25, 1.05)

    fig.tight_layout()
    plt.show()
