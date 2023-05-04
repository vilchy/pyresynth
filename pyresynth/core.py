"""Core functionality including loading audio from disk, computing envelope, generating sounds."""
from dataclasses import dataclass
from math import ceil, floor, inf
from types import NotImplementedType
from typing import List, Optional, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import FileDescriptorOrPath

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.io.wavfile
import scipy.signal
import sounddevice

from pyresynth import utils


@dataclass
class Axis:
    """A class for representing an axis.

    Constructor parameters:
        step: Spacing between values.
        start: Start value. The default value is 0.
    """

    step: float
    start: float = 0

    def range(self, length: int) -> npt.NDArray:
        """Return NumPy array of values forming an axis."""
        stop = self.start + length * self.step
        return np.linspace(self.start, stop, length, endpoint=False)

    def index(self, val: float) -> int:
        """Return zero-based index of a value."""
        return int(round((val - self.start) / self.step, 0))  # np.round returns float

    def __getitem__(self, index: int) -> float:
        """Return value for a given index."""
        return self.start + index * self.step


class Envelope:
    """A class to represent signal amplitude envelopes in the time domain."""

    def __init__(self, data: npt.NDArray, t_axis: Axis, threshold: Optional[float] = None) -> None:
        """`Envelope` constructor.

        :param data: NumPy array with the data.
        :param t_axis: Time axis.
        :param threshold: Minimum value in dBFS. The default value is the lowest value
            in the data or -90.0, whichever is the lower.
        """
        self.data = data
        self.t_axis = t_axis

        if threshold is None:
            self.threshold = np.min(data[data > -inf], initial=-90.0)
        else:
            self.threshold = threshold
        if len(data) > 0:
            # trim envelope to values only above the threshold
            above_threshold_indices = np.nonzero(data >= self.threshold)[0]
            # leave one sample of silence if possible
            first_idx = max(0, above_threshold_indices[0] - 1)
            last_idx = min(len(data) - 1, above_threshold_indices[-1] + 1)
            self.data = data[first_idx: last_idx + 1]
            self.t_axis.start += first_idx * self.t_axis.step  # time correction
        else:
            self.data = data

    def find_ranges_above_threshold(self, threshold: float = -80.0) \
            -> List[Tuple[float, float]]:
        """Find time ranges where the envelope is above a given threshold.

        :param threshold: Threshold in dBFS. The default is -80dBFS.
        :return: List of time ranges (in seconds) where the envelope is above the threshold.
            Format: [(start_time, end_time)].
        """
        above_threshold_data = (self.data > threshold).astype('int')
        threshold_transitions = above_threshold_data[1:] - above_threshold_data[:-1]
        start_indices = np.nonzero(threshold_transitions == 1)[0] + 1
        end_indices = np.nonzero(threshold_transitions == -1)[0] + 1
        if len(above_threshold_data) > 0:
            if above_threshold_data[0] == 1:
                # append, because no transition at the beginning
                start_indices = np.append([0], start_indices)
            if above_threshold_data[-1] == 1:
                # append, because no transition at the end
                end_indices = np.append(end_indices, [len(above_threshold_data)])

        return [(self.t_axis[start], self.t_axis[end])
                for start, end in zip(start_indices, end_indices)]

    def plot(self) -> None:
        """Plot the data using Matplotlib."""
        t_array = self.t_axis.range(len(self.data))
        plt.plot(t_array, self.data)
        plt.xlabel('t (sec)')
        plt.show()


class Sample:
    """A class to represent and manipulate a sound sample in the time domain."""

    default_sample_rate = 44100

    def __init__(self, data: Optional[npt.NDArray] = None,
                 sample_rate: int = default_sample_rate) -> None:
        """`Sample` constructor.

        :param data: Optional NumPy array with float32 data in the range [-1.0, 1.0].
        :param sample_rate: Optional sample rate of the sample.
        """
        if data is None:
            self.data = np.empty(0, dtype='float32')
        else:
            self.data = data
        self.sample_rate = sample_rate
        self.t_axis = Axis(step=1/sample_rate)

    @classmethod
    def load(cls, filename: 'FileDescriptorOrPath') -> 'Sample':
        """Load data from a WAV file and return `Sample` object.

        :param filename: Input WAV file.
        :return: `Sample` object with data read from a WAV file.
        """
        sample_rate, data = scipy.io.wavfile.read(filename)
        if data.ndim > 1:
            # Use only first channel if multichannel WAV
            data = data[:, 0]

        return cls(utils.normalize_wavdata(data), sample_rate)

    def save(self, filename: 'FileDescriptorOrPath') -> None:
        """Save `Sample` data to a WAV file.

        :param filename: Output WAV file (string or open file handle).
        """
        norm_data = utils.normalize_wavdata(self.data)
        return scipy.io.wavfile.write(filename, self.sample_rate, norm_data)

    @classmethod
    def generate_sin(cls, frequency: float, duration: float,
                     sample_rate: int = default_sample_rate) -> 'Sample':
        """Return a periodic sine waveform.

        :param frequency: Frequency (Hz) of the waveform.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the sine wave.
        """
        t_array = _t_array(duration, sample_rate)
        data = np.sin(2 * np.pi * frequency * t_array)
        return cls(data, sample_rate)

    @classmethod
    def generate_square(cls, frequency: float, duration: float,
                        sample_rate: int = default_sample_rate) -> 'Sample':
        """Return a periodic square-wave waveform.

        :param frequency: Frequency (Hz) of the waveform.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: A `Sample` object containing the square-wave waveform.
        """
        t_array = _t_array(duration, sample_rate)
        data = scipy.signal.square(2 * np.pi * frequency * t_array)
        return cls(data, sample_rate)

    @classmethod
    def generate_chirp(cls, frequency_0: float, frequency_1: float, duration: float,
                       sample_rate: int = default_sample_rate) -> 'Sample':
        """Frequency-swept cosine generator.

        :param frequency_0: Frequency (Hz) at time t=0.
        :param frequency_1: Frequency (Hz) at time t=duration.
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the signal with the requested time-varying frequency.
        """
        t_array = _t_array(duration, sample_rate)
        data = scipy.signal.chirp(t_array, f0=frequency_0, f1=frequency_1, t1=duration)
        return cls(data, sample_rate)

    @classmethod
    def generate_white_noise(cls, intensity: float, duration: float,
                             sample_rate: int = default_sample_rate) -> 'Sample':
        """Return a `Sample` with uniform white noise over [-intensity, intensity).

        :param intensity: Value range of the noise signal (maximal value should be 1.0).
        :param duration: Duration of the Sample in seconds.
        :param sample_rate: Sample rate in samples/sec.
        :return: `Sample` object containing the white noise signal.
        """
        data = np.random.uniform(low=-intensity, high=intensity,
                                 size=round(duration * sample_rate))
        return cls(data, sample_rate)

    @property
    def duration(self) -> float:
        """Return duration of the `Sample`.

        :return: `Sample` duration in seconds.
        """
        return len(self.data) / self.sample_rate

    def play(self) -> None:
        """Play back a NumPy array containing the audio data."""
        sounddevice.play(self.data, self.sample_rate)

    def split(self, threshold: float = -80.0) -> List['Sample']:
        """Split sounds separated by silence into individual samples.

        :param threshold: Threshold in dBFS.
        :return: List of `Sample` objects.
        """
        window_length = 1024
        envel = self.envelope_peak(window_length)
        ranges = envel.find_ranges_above_threshold(threshold)

        sample_list = []
        for start, end in ranges:
            data_slice = self.data[self.t_axis.index(start): self.t_axis.index(end)]
            sample_list.append(Sample(data_slice, self.sample_rate))
        return sample_list

    def envelope_peak(self, window_length: int, overlap: int = 0) -> Envelope:
        """Return envelope of peak amplitude values.

        :param window_length: Should be >= T/2 for a symmetric signal with fundamental period T.
        :param overlap: Percent of a window length to overlap.
        :return: Envelope of peak amplitude values.
        """
        def peak_func(array):
            return 20 * np.log10(np.max(np.abs(array)))
        return self.__envelope(peak_func, window_length, overlap)

    def envelope_rms(self, window_length: int, overlap: int = 0) -> Envelope:
        """Return RMS (Root Mean Square) amplitude envelope of the data.

        :param window_length: Should be >= T/2 for a symmetric signal with fundamental period T.
        :param overlap: Percent of a window to overlap.
        :return: Envelope of RMS amplitude values.
        """
        def rms_func(array):
            return 10 * np.log10(np.mean(np.square(array)))
        return self.__envelope(rms_func, window_length, overlap)

    def __envelope(self, fun, window_length, overlap=0):
        hop_length = floor(window_length * (100 - overlap) / 100)
        frame_count = ceil((len(self.data) - window_length) / hop_length) + 1
        if frame_count < 1:
            frame_count = 1
        envelope_data = np.zeros(frame_count)

        for i in range(0, frame_count):
            # (last frame will be shorter)
            envelope_data[i] = fun(self.data[i * hop_length: i * hop_length + window_length])

        envelope_step = hop_length * self.t_axis.step
        envelope_start = window_length / 2 * self.t_axis.step
        return Envelope(envelope_data, Axis(envelope_step, envelope_start))

    def __add__(self, other: object) -> 'Sample':
        """Return self+other. Works only if sample rates match."""
        if isinstance(other, (int, float)):
            return Sample(self.data + other, self.sample_rate)
        elif isinstance(other, Sample):
            return self.__binary_op(other, lambda x, y: x + y, lambda x: x)
        return NotImplemented

    def __sub__(self, other: object) -> 'Sample':
        """Return self-other. Works only if sample rates match."""
        if isinstance(other, (int, float)):
            return Sample(self.data - other, self.sample_rate)
        elif isinstance(other, Sample):
            return self.__binary_op(other, lambda x, y: x - y, lambda x: -x)
        return NotImplemented

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

    def __mul__(self, other: object) -> 'Sample | NotImplementedType':
        """Return self*other. Works only for multiplication by a scalar."""
        if isinstance(other, (int, float)):
            return Sample(self.data * other, self.sample_rate)
        return NotImplemented

    def __rmul__(self, other: object) -> 'Sample | NotImplementedType':
        """Return other*self. Works only for multiplication by a scalar."""
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def plot(self) -> None:
        """Plot the data using Matplotlib."""
        t_array = self.t_axis.range(len(self.data))
        plt.plot(t_array, self.data)
        plt.xlabel('t (sec)')
        plt.show()


def _t_array(duration, sample_rate):
    return np.linspace(0, duration, round(duration * sample_rate), endpoint=False)


@dataclass
class TimeFrequency:
    """A class to represent a sound sample in the time-frequency domain."""

    spectrum: npt.NDArray
    phase: npt.NDArray
    t_axis: Axis
    f_axis: Axis

    @classmethod
    def stft(cls, sample: Sample, window_length: int = 2047, fft_length: int = 8192,
             window_type: str | float | tuple = 'blackman', overlap: int = 0) -> 'TimeFrequency':
        """Return Time-frequency representation using Short-time Fourier transform.

        :param sample: Input `Sample`.
        :param window_length: Length of a window function.
        :param overlap: Percent of a window to overlap.
        :param fft_length: Transform length (most efficient for power of 2).
        :param window_type: Window function type as in scipy.signal.get_window.
        :return: `TimeFrequency` representation of the input sample.
        """
        window = scipy.signal.get_window(window_type, window_length, False)
        coherent_power_gain = 20 * np.log10(window_length / sum(window))

        hop_length = floor(window_length * (100 - overlap) / 100)
        frame_count = ceil((len(sample.data) - window_length) / hop_length) + 1
        width = floor(fft_length / 2) + 1  # only real input
        spectrum_array = np.zeros([frame_count, width])
        phase_array = np.zeros([frame_count, width])

        for i in range(0, frame_count - 1):
            data_slice = sample.data[i * hop_length: i * hop_length + window_length]
            spectrum, phase = _stft_frame(data_slice, fft_length, window, coherent_power_gain)
            spectrum_array[i, :] = spectrum
            phase_array[i, :] = phase
        # last frame
        data_slice = sample.data[(frame_count - 1) * hop_length:]
        window = scipy.signal.get_window(window_type, len(data_slice), False)
        spectrum, phase = _stft_frame(data_slice, fft_length, window, coherent_power_gain)
        spectrum_array[frame_count - 1, :] = spectrum
        phase_array[frame_count - 1, :] = phase

        t_step = hop_length / sample.sample_rate
        t_start = window_length / (2 * sample.sample_rate)
        f_step = sample.sample_rate / fft_length
        return cls(spectrum_array, phase_array, Axis(t_step, t_start), Axis(f_step))

    def plot_spectrogram(self) -> None:
        """Plot the spectrogram using Matplotlib."""
        values = np.transpose(self.spectrum[:, :])
        t_array = self.t_axis.range(self.spectrum.shape[0])
        f_array = self.f_axis.range(self.spectrum.shape[1])

        T, F = np.meshgrid(t_array, f_array)
        plt.pcolormesh(T, F, values, shading='gouraud')
        plt.show()


def _stft_frame(in_array, fft_length, window, coherent_power_gain):
    ft_result = _zero_phase_rfft(in_array, fft_length, window)
    spectrum = abs(ft_result) / len(window)
    spectrum[1:] *= 2  # single-sided FT requires multiplication by 2
    with np.errstate(divide='ignore'):
        log_power_spectrum = 20 * np.log10(spectrum) + coherent_power_gain
    phase = np.unwrap(np.angle(ft_result))
    return log_power_spectrum, phase


def _zero_phase_rfft(in_array, fft_length, window):
    windowed = in_array * window
    # zero phase padding
    window_mid = floor(len(window)/2)
    fft_input = np.zeros(fft_length)
    fft_input[:ceil(len(window) / 2)] = windowed[window_mid:]
    fft_input[-floor(len(window) / 2):] = windowed[:window_mid]
    return np.fft.rfft(fft_input)
