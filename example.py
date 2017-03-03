## imports
import imgtosnd as i2s
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

## short-time fourier parameters
sample_rate = 16000  # sample rate in Hz
frame_size = i2s.next_pow2(sample_rate*25/1000)  # frame size in samples
hop_size = int(frame_size/2)  # hop size (samples)

## read image and get complex spectrogram with random phase
im = i2s.read_image_as_spec("data/me.jpg", frame_size, sample_rate, hop_size, num_seconds=20)
im **= 2
im = i2s.add_random_phase(im)

## convert image to audio via inverse short-time fft
x = i2s.st_irfft(im, frame_size, hop_size, win_sqrt=False)

## apply some normalization to the audio signal and write to disk
x /= np.max(np.abs(x))
wavfile.write('data\\me.wav', sample_rate, x)

## get back spectrogram of synthesized waveform
xf = i2s.st_rfft(x, frame_size, hop_size)
xf = np.abs(xf)
xf /= np.max(xf)

## plot stuff
ax = plt.subplot(121)
t = np.arange(np.size(x))/sample_rate
plt.plot(t, x)
plt.grid()
plt.xlabel('Time (s)')

plt.subplot(122, sharex=ax)
t = np.arange(np.shape(im)[1])/(sample_rate/hop_size)
f = np.fft.rfftfreq(frame_size)*sample_rate/1000
plt.pcolormesh(t, f, 20*np.log10(xf), cmap='plasma')
plt.colorbar(label='level(dB)')
plt.clim([-30, 0])
plt.colormaps()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')

plt.gcf().autofmt_xdate()