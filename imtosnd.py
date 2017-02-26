##
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sig

##
im = Image.open("me.jpg").convert(mode='F')
im = np.asarray(im)
im = im / np.max(im)
plt.imshow(im)


##
def next_pow2(x):
    return np.int(np.exp2(np.round(np.log2(x))))


def st_rfft(x, frame_size, hop_size, fft_size):
    idx_starts = np.arange(0, len(x)-frame_size, hop_size, dtype='int')
    xf = np.zeros([len(idx_starts), int(fft_size/2+1)], dtype=np.complex)
    win = np.sqrt(sig.hann(frame_size))

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.rfft(x[idx_start:idx_stop]*win, n=fft_size)
        xf[cnt, :] = xtemp

    return xf


def st_irfft(xf, frame_size, hop_size, fft_size):
    lenx = len(xf)*hop_size + frame_size
    x = np.zeros(lenx)
    win = np.sqrt(sig.hann(frame_size))
    idx_starts = np.arange(0, len(x) - frame_size, hop_size, dtype='int')

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.irfft(xf[cnt, :])
        x[idx_start:idx_stop] += xtemp[:frame_size]*win

    return x

# generate a sweep
fs = 8000
N = fs * 5
n = np.arange(N)
f = (fs/4)*np.arange(N)/N
x = np.cos(f*2*np.pi*n/fs)


sd.play(x, samplerate=fs)

## get spectrogram
frame_size = next_pow2(fs*25/1000)
s = {'frame_size': frame_size, 'hop_size': int(frame_size/2), 'fft_size': int(frame_size*2)}
xf = st_rfft(x, **s)
xx = st_irfft(xf, **s)

# plt.plot(x)
plt.plot(xx-x)


##
xf = st_rfft(x, **s)
xx = st_irfft(xf, **s)



xf = np.abs(xf)

plt.pcolor(xf)
plt.colorbar()

##
import scipy

