import numpy as np

from pyresynth import core


def find_ranges(test_list, threshold):
    e = core.Envelope(np.array(test_list), core.Axis(step=1))
    return list(e.find_ranges_above_threshold(threshold))


def test_find_ranges_above_threshold_for_empty():
    assert find_ranges([], -80) == []


def test_find_ranges_above_threshold_for_silence():
    assert find_ranges([-90, -83, -91], -80) == []


def test_find_ranges_above_threshold_for_single_padded_range():
    assert find_ranges([-90, -83, -10, -9, -91], -80) == [(2, 4)]


def test_find_ranges_above_threshold_for_single_not_padded_range():
    assert find_ranges([-9, -8, -10, -9, -9], -80) == [(0, 5)]


def test_find_ranges_above_threshold_for_double_not_padded_range():
    assert find_ranges([-9, -8, -10, -89, -9], -80) == [(0, 3), (4, 5)]


def envelope_data(test_list, threshold):
    return list(core.Envelope(np.array(test_list), core.Axis(step=0.5), threshold).data)


def test_envelope_for_all_values_above_threshold():
    assert envelope_data([-80, -39, -43, -32, -80], -90) == [-80, -39, -43, -32, -80]


def test_envelope_for_values_below_threshold():
    assert envelope_data([-180, -39, -43, -32, -180], -90) == [-180, -39, -43, -32, -180]
    assert envelope_data([-180, -180, -39, -43, -32, -180, -180], -90) \
        == [-180, -39, -43, -32, -180]


def test_envelope_peak_value():
    sin_signal = core.Sample.generate_sin(1, 1, 100)
    sin_peak_value = sin_signal.envelope_peak(100).data[0]
    np.testing.assert_almost_equal(sin_peak_value, 0, 2)

    square_signal = core.Sample.generate_sin(1, 1, 100)
    square_peak_value = square_signal.envelope_peak(100).data[0]
    np.testing.assert_almost_equal(square_peak_value, 0, 2)


def test_envelope_rms_value():
    sin_signal = core.Sample.generate_sin(1, 1, 100)
    sin_rms_value = sin_signal.envelope_rms(100).data[0]
    np.testing.assert_almost_equal(sin_rms_value, -3, 2)

    square_signal = core.Sample.generate_sin(1, 1, 100)
    square_rms_value = square_signal.envelope_peak(100).data[0]
    np.testing.assert_almost_equal(square_rms_value, 0, 2)


def test_sample_add_for_samples_with_equal_lengths():
    s1 = core.Sample(np.array([0.2, 0.3]))
    s2 = core.Sample(np.array([0.5, -0.7]))
    np.testing.assert_almost_equal((s1 + s2).data, np.array([0.7, -0.4]))


def test_sample_add_for_samples_with_unequal_lengths():
    s1 = core.Sample(np.array([0.2, 0.3]))
    s2 = core.Sample(np.array([0.5]))
    np.testing.assert_almost_equal((s1 + s2).data, np.array([0.7, 0.3]))

    s1 = core.Sample(np.array([0.2]))
    s2 = core.Sample(np.array([0.5, -0.7]))
    np.testing.assert_almost_equal((s1 + s2).data, np.array([0.7, -0.7]))


def test_sample_add_for_sample_and_scalar():
    s = core.Sample(np.array([0.2, 0.3]))
    np.testing.assert_almost_equal((s + 0.5).data, np.array([0.7, 0.8]))


def test_sample_sub_for_samples_with_equal_lengths():
    s1 = core.Sample(np.array([0.2, 0.3]))
    s2 = core.Sample(np.array([0.5, -0.7]))
    np.testing.assert_almost_equal((s1 - s2).data, np.array([-0.3, 1.0]))


def test_sample_sub_for_samples_with_unequal_lengths():
    s1 = core.Sample(np.array([0.2, 0.3]))
    s2 = core.Sample(np.array([0.5]))
    np.testing.assert_almost_equal((s1 - s2).data, np.array([-0.3, 0.3]))

    s1 = core.Sample(np.array([0.2]))
    s2 = core.Sample(np.array([0.5, -0.7]))
    np.testing.assert_almost_equal((s1 - s2).data, np.array([-0.3, 0.7]))


def test_sample_sub_for_sample_and_scalar():
    s = core.Sample(np.array([0.2, 0.3]))
    np.testing.assert_almost_equal((s - 0.5).data, np.array([-0.3, -0.2]))


def test_time_frequency_stft_spectrum_shape():
    # sanity check
    s = core.Sample(np.ones(16))

    tf = core.TimeFrequency.stft(s,  window_length=4, fft_length=4)
    assert np.shape(tf.spectrum) == (4, 3)

    tf = core.TimeFrequency.stft(s,  window_length=4, fft_length=8)
    assert np.shape(tf.spectrum) == (4, 5)

    tf = core.TimeFrequency.stft(s,  window_length=4, fft_length=4, overlap=50)
    assert np.shape(tf.spectrum) == (7, 3)
