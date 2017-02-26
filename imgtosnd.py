## imports
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sig

## not used
def next_pow2(x):
    return np.int(np.exp2(np.round(np.log2(x))))


def sweep(num_seconds, sample_rate):
    N = sample_rate * num_seconds
    n = np.arange(N)
    f = (sample_rate / 4) * np.arange(N) / N
    x = np.cos(f * 2 * np.pi * n / sample_rate)
    return x


def st_rfft(x, frame_size, hop_size, fft_size):
    idx_starts = np.arange(0, len(x)-frame_size, hop_size, dtype='int')
    xf = np.zeros([len(idx_starts), int(fft_size/2+1)], dtype=np.complex)
    win = np.sqrt(sig.hann(frame_size, False))

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.rfft(x[idx_start:idx_stop]*win, n=fft_size)
        xf[cnt, :] = xtemp

    return xf


def st_irfft(xf, frame_size, hop_size, win_sqrt=False, apply_fft_shift=False):
    lenx = len(xf)*hop_size + frame_size
    x = np.zeros(lenx)
    win = sig.hann(frame_size, False)
    if win_sqrt:
        win = np.sqrt(win)
    idx_starts = np.arange(0, len(x) - frame_size, hop_size, dtype='int')

    if apply_fft_shift:
        fftshift = np.fft.ifftshift
    else:
        fftshift = lambda x: x

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.irfft(xf[cnt, :])
        x[idx_start:idx_stop] += fftshift(xtemp[:frame_size])*win

    return x

## get image and convert pixels to grayscale normalized between [0, 1]
im = Image.open("data/me.jpg").convert(mode='F')
im = np.asarray(im)
im.setflags(write=True)
im -= np.min(im)
im /= np.max(im)

## audio parameters
sample_rate = 8000
frame_size = next_pow2(sample_rate*25/1000)
hop_size = int(frame_size/2)
fft_size = int(frame_size*4)

## convert image to audio via inverse short-time fft
im_t = st_irfft(np.flipud(im), frame_size, hop_size, win_sqrt=False, apply_fft_shift=True)
im_t /= np.max(im_t)
xf = st_rfft(im_t, frame_size, hop_size, fft_size)
xf = np.abs(xf)

## show spectrogram
plt.pcolor(xf)
plt.colorbar()

## play sound
sd.play(im_t, samplerate=sample_rate)
