import collections
from dataclasses import dataclass, field

from typing import Collection, Callable, Optional, List, Deque

import numpy as np
import pandas as pd
import scipy.signal
from loguru import logger
from numpy.typing import NDArray

dt = 1e-3


# default: 100 Mbit/s (Fast Ethernet)
def tx_delay(bytes: int, bps=100_000_000) -> float:
    return bytes * 8 / bps / dt


def autocorrelation(x: NDArray[float]) -> NDArray[float]:
    return (scipy.signal.correlate(x - x.mean(), x - x.mean(), mode='full') / len(x) / x.var())[len(x) - 1:]


@dataclass
class Packet:
    src_ip: str
    timestamp: float
    bytes: int


@dataclass
class Packets:
    packets: Collection[Packet]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([p.__dict__ for p in self.packets])

    def __iter__(self):
        return iter(self.packets)

    def __len__(self):
        return len(self.packets)


@dataclass
class Correlation:
    name: str = field(init=False)
    tau0: float
    sigma: float

    def __call__(self, *args, **kwargs):
        pass

    def generate(self, noise: NDArray[float]) -> NDArray[float]:
        pass


@dataclass
class TriangularCorr(Correlation):
    name = 'triangular'

    def __call__(self, tau: NDArray[float]) -> NDArray[float]:
        return np.where(
            abs(tau) <= self.tau0,
            self.sigma ** 2 * (1 - (abs(tau) / self.tau0)),
            0
        )

    def generate(self, noise: NDArray[float]) -> NDArray[float]:
        L = len(noise)
        # N = min(round(self.tau0 / dt), L)
        N = min(round(self.tau0 / 1), L)
        signal = np.zeros(L)

        for n in range(L):
            sum = 0
            for k in range(1, N):
                sum += noise[n - k] / N
            signal[n] = self.sigma * np.sqrt(N) * sum

        return signal


@dataclass
class ExponentialCorr(Correlation):
    name = 'exponential'

    def __call__(self, tau: NDArray[float]) -> NDArray[float]:
        return self.sigma ** 2 * np.exp(-abs(tau) / self.tau0)

    def generate(self, noise: NDArray[float]) -> NDArray[float]:
        L = len(noise)
        # gamma = dt / self.tau0
        gamma = 1 / self.tau0
        rho = np.exp(-gamma)
        b1 = rho
        a0 = self.sigma * np.sqrt(1 - rho ** 2)
        signal = np.zeros(L)

        for n in range(1, L):
            signal[n] = a0 * noise[n] + b1 * signal[n - 1]

        return signal


@dataclass(init=True)
class OnOffSource:
    src_ip: str
    data_rate: int
    packet_size: Callable[[], float]
    on_time: Callable[[], float]
    off_time: Callable[[], float]
    max_bytes: Optional[int]

    _result: Packets = None
    on_durations: List[float] = field(default_factory=list)
    off_durations: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.max_bytes is None:
            self.max_bytes = np.inf

    def start(self) -> Packets:
        if self._result:
            return self._result

        logger.debug('start {}', self)
        timestamp = 0
        sent_bytes = 0
        result = []
        try:
            while sent_bytes < self.max_bytes:
                off_time = self.off_time()
                assert off_time >= 0, "off_time is negative: " + str(off_time)
                on_time = self.on_time()
                assert on_time >= 0, "on_time is negative: " + str(on_time)
                logger.debug('off: {}, on: {}', off_time, on_time)
                self.off_durations.append(off_time)
                self.on_durations.append(on_time)
                timestamp += off_time
                on_duration = 0
                sent_before = len(result)
                while on_duration < on_time and sent_bytes < self.max_bytes:
                    packet_bytes = int(self.packet_size())
                    delay = tx_delay(packet_bytes, self.data_rate)
                    # if on_duration + delay > on_time:
                    #     break
                    result.append(Packet(self.src_ip, timestamp, packet_bytes))
                    sent_bytes += packet_bytes
                    on_duration += delay
                    timestamp += delay
                sent_after = len(result)
                logger.debug('sent {} packets during on period', sent_after - sent_before)
        except StopIteration:
            logger.debug('iterator-based on/off generator stopped')

        self._result = Packets(result)
        return self._result


def _statistic_update(statistic: List[int], diff: int):
    if len(statistic) == 0:
        statistic.append(diff)
    else:
        statistic.append(statistic[-1] + diff)


@dataclass
class Multiplexer:
    sources: List[OnOffSource]
    data_rate: int
    queue: Deque[Packet] = collections.deque()
    out: Deque[Packet] = collections.deque()

    _result: Packets = None
    queue_len_count: List[int] = field(default_factory=list)
    queue_len_bytes: List[int] = field(default_factory=list)

    def last_transmitted(self) -> Optional[Packet]:
        return self.out[-1] if self.out else None

    def enqueue(self, p: Packet):
        self.queue.append(p)
        _statistic_update(self.queue_len_count, +1)
        _statistic_update(self.queue_len_bytes, +p.bytes)

    def transmit(self, p: Packet):
        self.out.append(p)
        _statistic_update(self.queue_len_count, -1)
        _statistic_update(self.queue_len_bytes, -p.bytes)

    def drain_queue(self):
        while len(self.queue) > 0:
            queued = self.queue.popleft()
            last = self.last_transmitted()
            if last and (last.timestamp + tx_delay(last.bytes, self.data_rate)) >= queued.timestamp:
                queued.timestamp = last.timestamp + tx_delay(last.bytes, self.data_rate)
            self.transmit(queued)

    def start(self) -> Packets:
        if self._result:
            return self._result

        data = pd.DataFrame()
        for priority, source in enumerate(self.sources):
            df = source.start().to_dataframe()
            df['priority'] = priority
            data = data.append(df)
        data = data.sort_values(['timestamp', 'priority'])
        del data['priority']

        for value in data.values:
            p = Packet(*value)
            last = self.last_transmitted()

            if not last:
                self.out.append(p)
            elif (last.timestamp + tx_delay(last.bytes, self.data_rate)) < p.timestamp:
                if len(self.queue) > 0:
                    self.enqueue(p)
                    self.drain_queue()
                else:
                    self.out.append(p)
            else:
                self.enqueue(p)

        self.drain_queue()
        self._result = Packets(self.out)
        return self._result
