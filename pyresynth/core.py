"""Core functionality including loading audio from disk, computing envelope, generating sounds."""
from dataclasses import dataclass
from math import ceil, floor, inf
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice

from pyresynth import utils


@dataclass
class DataOverTime:
    """Base class for representing NumPy data with a time axis.

    Constructor parameters:
        data: NumPy array with the data.
        delta_t: Spacing between time values (in seconds).
        t_0: Start time in seconds. The default value is 0.
    """

    data: np.ndarray
    delta_t: float
    t_0: float = 0

    @property
    def t_axis(self) -> np.ndarray:
        """Return NumPy array of time values forming a time axis."""
        stop = self.t_0 + len(self.data) * self.delta_t
        return np.linspace(self.t_0, stop, len(self.data), endpoint=False)

    def plot(self):
        """Plot the data using Matplotlib."""
        plt.plot(self.t_axis, self.data)
        plt.xlabel('t (sec)')
        plt.show()

    def t_to_idx(self, time: float) -> int:
        """Convert a time value to a position in the data array.

        :param time: Time in seconds.
        :return: Position of the time value.
        """
        return int(round((time - self.t_0) / self.delta_t, 0))  # np.round returns float

    def idx_to_t(self, idx: int) -> float:
        """Convert position in the data array to a time value.

        :param idx: Position in the data array.
        :return: Time value (in seconds) of the position in the data array.
        """
        return self.t_0 + idx * self.delta_t


class Envelope(DataOverTime):
    """A class to represent signal amplitude envelopes in the time domain."""

    def __init__(self, data: np.ndarray, delta_t: float, t_0: float = 0, threshold=None):
        """`Envelope` constructor.

        :param data: NumPy array with the data.
        :param delta_t: Spacing in time between values.
        :param t_0: Start time. The default value is 0.
        :param threshold: Minimum value in dBFS. The default value is the lowest value
            in the data or -90, whichever is the lower.
        """
        super().__init__(data=data, delta_t=delta_t, t_0=t_0)
        if threshold is None:
            self.threshold = np.min(data[data > -inf], initial=-90)
        else:
            self.threshold = threshold
        if len(data) > 0:
            # trim envelope to values only above the threshold
            above_threshold_indices = np.where(data >= self.threshold)[0]
            # leave one sample of silence if possible
            first_idx = max(0, above_threshold_indices[0] - 1)
            last_idx = min(len(data) - 1, above_threshold_indices[-1] + 1)
            self.data = data[first_idx: last_idx + 1]
            self.t_0 += first_idx * self.delta_t  # time correction
        else:
            self.data = data

    def find_ranges_above_threshold(self, threshold: int = -80) \
            -> List[Tuple[float, float]]:
        """Find time ranges where the envelope is above a given threshold.

        :param threshold: Threshold in dBFS. The default is -80dBFS.
        :return: List of time ranges (in seconds) where the envelope is above the threshold.
            Format: [(start_time, end_time)].
        """
        above_threshold_data = (self.data > threshold).astype('int')
        threshold_transitions = above_threshold_data[1:] - above_threshold_data[:-1]
        start_indices = np.where(threshold_transitions == 1)[0] + 1
        end_indices = np.where(threshold_transitions == -1)[0] + 1
        if len(above_threshold_data) > 0:
            if above_threshold_data[0] == 1:
                # append, because no transition at the beginning
                start_indices = np.append([0], start_indices)
            if above_threshold_data[-1] == 1:
                # append, because no transition at the end
                end_indices = np.append(end_indices, [len(above_threshold_data)])

        return [(self.idx_to_t(start), self.idx_to_t(end))
                for start, end in zip(start_indices, end_indices)]


class Sample(DataOverTime):
    """A class to represent and manipulate a sound sample in the time domain."""

    default_sample_rate = 44100

    def __init__(self, data: Optional[np.ndarray] = None, sample_rate: int = default_sample_rate):
        """`Sample` constructor.

        :param data: Optional NumPy array with float32 data in the range [-1.0, 1.0].
        :param sample_rate: Optional sample rate of the sample.
        """
        if data is None:
            self.data = np.empty(0, dtype='float32')
        else:
            self.data = data
        self.sample_rate = sample_rate
        self.delta_t = 1/sample_rate
        self.t_0 = 0

    @classmethod
    def load(cls, filename) -> 'Sample':
        """Load data from a WAV file and return `Sample` object.

        :param filename: Input WAV file.
        :return: `Sample` object with data read from a WAV file.
        """
        sample_rate, data = scipy.io.wavfile.read(filename)
        if data.ndim > 1:
            # Use only first channel if multichannel WAV
            data = data[:, 0]

        return cls(utils.normalize_wavdata(data), sample_rate)

    def save(self, filename):
        """Save `Sample` data to a WAV file.

        :param filename: Output WAV file (string or open file handle).
        """
        norm_data = utils.normalize_wavdata(self.data)
        return scipy.io.wavfile.write(filename, self.sample_rate, norm_data)

    @classmethod
    def generate_sin(cls, frequency: float, duration: float,
                     sample_rate: int = default_sample_rate):
        """Return a periodic sine waveform.

        :param frequency: Frequency (Hz) of the waveform.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the sine wave.
        """
        t_array = np.linspace(0, duration, round(duration * sample_rate), endpoint=False)
        data = np.sin(2 * np.pi * frequency * t_array)
        return cls(data, sample_rate)

    @classmethod
    def generate_square(cls, frequency: float, duration: float,
                        sample_rate: int = default_sample_rate):
        """Return a periodic square-wave waveform.

        :param frequency: Frequency (Hz) of the waveform.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: A `Sample` object containing the square-wave waveform.
        """
        t_array = np.linspace(0, duration, round(duration * sample_rate), endpoint=False)
        data = scipy.signal.square(2 * np.pi * frequency * t_array)
        return cls(data, sample_rate)

    @classmethod
    def generate_chirp(cls, frequency_0: float, frequency_1: float, duration: float,
                       sample_rate: int = default_sample_rate):
        """Frequency-swept cosine generator.

        :param frequency_0: Frequency (Hz) at time t=0.
        :param frequency_1: Frequency (Hz) at time t=duration.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the signal with the requested time-varying frequency.
        """
        t_array = np.linspace(0, duration, round(duration * sample_rate), endpoint=False)
        data = scipy.signal.chirp(t_array, f0=frequency_0, f1=frequency_1, t1=duration)
        return cls(data, sample_rate)

    @classmethod
    def generate_white_noise(cls, intensity: float, duration: float,
                             sample_rate: int = default_sample_rate):
        """Return a `Sample` with uniform white noise over [-intensity, intensity).

        :param intensity: Value range of the noise signal (maximal value should be 1.0).
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the white noise signal.
        """
        data = np.random.uniform(low=-intensity, high=intensity, size=round(duration*sample_rate))
        return cls(data, sample_rate)

    @property
    def duration(self) -> float:
        """Return duration of the `Sample`.

        :return: `Sample` duration in seconds.
        """
        return len(self.data) / self.sample_rate

    def play(self):
        """Play back a NumPy array containing the audio data."""
        sounddevice.play(self.data, self.sample_rate)

    def split(self, threshold: int = -80) -> List['Sample']:
        """Split sounds separated by silence into individual samples.

        :param threshold: Threshold in dBFS.
        :return: List of `Sample` objects.
        """
        window_length = 1024
        envel = self.envelope_peak(window_length)
        ranges = envel.find_ranges_above_threshold(threshold)
        return [Sample(self.data[self.t_to_idx(start): self.t_to_idx(end)], self.sample_rate)
                for start, end in ranges]

    def envelope_peak(self, window_length: int, overlap: int = 0) -> 'Envelope':
        """Return envelope of peak amplitude values.

        :param window_length: Should be >= T/2 for a symmetric signal with fundamental period T.
        :param overlap: Percent of a window length to overlap.
        :return: Envelope of peak amplitude values.
        """
        def peak_func(array):
            return 20 * np.log10(np.max(np.abs(array)))
        return self.__envelope(peak_func, window_length, overlap)

    def envelope_rms(self, window_length: int, overlap: int = 0) -> 'Envelope':
        """Return RMS (Root Mean Square) amplitude envelope of the data.

        :param window_length: Should be >= T/2 for a symmetric signal with fundamental period T.
        :param overlap: Percent of a window to overlap.
        :return: Envelope of RMS amplitude values.
        """
        def rms_func(array):
            return 10 * np.log10(np.mean(np.square(array)))
        return self.__envelope(rms_func, window_length, overlap)

    reduce_fun = Callable[[np.array], float]

    def __envelope(self, fun: reduce_fun, window_length: int, overlap: int = 0):
        hop_length = floor(window_length * (100 - overlap) / 100)
        frame_count = ceil((len(self.data) - window_length) / hop_length) + 1
        if frame_count < 1:
            frame_count = 1
        envelope_data = np.zeros(frame_count)

        for i in range(0, frame_count):
            # (last frame will be shorter)
            envelope_data[i] = fun(self.data[i * hop_length: i * hop_length + window_length])
        return Envelope(envelope_data, hop_length * self.delta_t, window_length / 2 * self.delta_t)

    def __sub__(self, other: 'Sample') -> 'Sample':
        """Return self-other. Works only if sample rates match."""
        return self.__binary_op(other, lambda x, y: x - y, lambda x: -x)

    def __add__(self, other: 'Sample') -> 'Sample':
        """Return self+other. Works only if sample rates match."""
        return self.__binary_op(other, lambda x, y: x + y, lambda x: x)

    def __binary_op(self, other, binary_fun, unary_fun):
        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rate mismatch.", self.sample_rate, other.sample_rate)

        min_length = min(len(self.data), len(other.data))
        max_length = max(len(self.data), len(other.data))
        new_data = np.empty(max_length)
        new_data[0:min_length] = binary_fun(self.data[0:min_length], other.data[0:min_length])

        if len(self.data) > len(other.data):
            new_data[min_length:] = self.data[min_length:]
        else:
            new_data[min_length:] = unary_fun(other.data[min_length:])

        return Sample(new_data, self.sample_rate)

    def __mul__(self, other):
        """Return self*other. Works only for multiplication by a scalar."""
        if isinstance(other, (int, float)):
            return Sample(self.data * other, self.sample_rate)
        return NotImplemented

    def __rmul__(self, other):
        """Return other*self. Works only for multiplication by a scalar."""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented
