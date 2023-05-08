pyresynth
=========
A python package for sound analysis.

Work in progress…

Installation
------------
You can download or clone the repository and use `pip` to handle dependencies:

```
$ unzip pyresynth.zip
$ pip install -e pyresynth
```
or
```
$ git clone https://github.com/vilchy/pyresynth.git
$ pip install -e pyresynth
```

Example use
-----------
```
from pyresynth import Sample, TimeFrequency

# read a WAV file and play it
song = Sample.load("the_liberty_bell.wav")
song.play()

# generate sin + noise signal
s1 = Sample.generate_sin(294, 1)  # 294 Hz for 1 second
s2 = Sample.generate_white_noise(1, 1)  # for 1 second
s3 = s1 * 0.5 + s2 * 0.25

# save the results to a WAV file
s3.save("out_signal.wav")

# plot RMS amplitude envelope
e = s3.envelope_rms(150)
e.plot()

# return time-frequency representation
tf = TimeFrequency.stft(s3)

# plot spectrogram
tf.plot_spectrogram()
```


Running tests
----------------
```
$ python3 -m pytest tests
```

