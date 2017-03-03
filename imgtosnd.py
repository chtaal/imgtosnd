from PIL import Image
import numpy as np
import scipy.signal as sig
from scipy import misc


def read_image_as_spec(pth, fft_size=512, sample_rate=8000, hop_size=256, num_seconds=5):
    """
    Read an image and convert into a 2d numpy array. The array can be correctly sized such that it can be used to
    audify the image by applying an inverse short-time Fourier transform.

    Parameters
    ----------
    pth : str
        path to image
    fft_size : int
        size of fourier window
    sample_rate : int
        samplerate (hz)
    hop_size : int
        number of samples used in short-time fft between two consecutive windows
    num_seconds : float
        length of the returned spectrogram (s)

    Returns
    -------
    narray
        image spectrogram as 2D numpy array

    """
    # open image and convert to floating point grayscale
    im = Image.open(pth).convert(mode='F')

    # resize to the desired fourier parameters
    im = misc.imresize(im, (int(fft_size / 2 + 1), int(num_seconds * (sample_rate / hop_size))), mode='F')

    # set write flag to true so we can change the values (not sure why this was needed)
    im.setflags(write=True)

    # apply some normalization
    im -= np.min(im)
    im /= np.max(im)

    # we flip it because images are mirrored vertically compared to spectrograms
    im = np.flipud(im)

    return im


def next_pow2(x):
    """
    get closest power of 2
    """
    return np.int(np.exp2(np.round(np.log2(x))))


def sweep(num_seconds, sample_rate):
    """
    Generate a sinusoidal sweep. Actually not used in this project, but I used it to confirm that st_rfft and
    st_irfft are working as expected.

    Parameters
    ----------
    num_seconds : int
        sweep length (s)
    sample_rate : int
        sample rate (Hz)

    Returns
    -------
    narray
        sweep
    """
    num_samples = sample_rate * num_seconds
    n = np.arange(num_samples)
    f = (sample_rate / 4) * np.arange(num_samples) / num_samples
    x = np.cos(f * 2 * np.pi * n / sample_rate)
    return x


def st_rfft(x, frame_size, hop_size, fft_size=None):
    """
    short-time real fast fourier transform. Will apply the rfft to short-time overlapping segments of the input
    signal.

    Parameters
    ----------
    x : np.array
        input signal
    frame_size : int
        length of the window (samples)
    hop_size : int
        time between two consecutive windows (samples)
    fft_size : int
        size of the fft window (samples)

    Returns
    -------
    narray
        2d np.array where its shape corresponds to (frequency bins, time frames)

    """
    if not fft_size:
        fft_size = frame_size
    idx_starts = np.arange(0, len(x)-frame_size, hop_size, dtype='int')
    xf = np.zeros([int(fft_size/2+1), len(idx_starts)], dtype=np.complex)
    win = np.sqrt(sig.hann(frame_size, False))

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.rfft(x[idx_start:idx_stop]*win, n=fft_size)
        xf[:, cnt] = xtemp

    return xf


def st_irfft(xf, frame_size, hop_size, win_sqrt=False):
    """
    inverse short-time real fast fourier transform.

    Parameters
    ----------
    xf : narray
        spectrogram
    frame_size : int
        frame size (samples)
    hop_size : int
        number of samples between two consecutive windows
    win_sqrt : bool
        apply square-rooted window. This is only relevant if your spectrogram is derived from st_rfft. Applying two
        square rooted windows twice will then result in a perfect overlapping window.

    Returns
    -------
    narray
        time-domain signal
    """
    lenx = xf.shape[1]*hop_size + frame_size
    x = np.zeros(lenx)
    win = sig.hann(frame_size, False)
    if win_sqrt:
        win = np.sqrt(win)
    idx_starts = np.arange(0, len(x) - frame_size, hop_size, dtype='int')

    for cnt, idx_start in enumerate(idx_starts):
        idx_stop = idx_start + frame_size
        xtemp = np.fft.irfft(xf[:, cnt])
        x[idx_start:idx_stop] += xtemp[:frame_size]*win

    return x


def add_random_phase(x):
    """
    add random complex phase [-pi, pi] to all values in x
    """
    return x*np.exp(1j * np.random.uniform(-np.pi, np.pi, x.shape))
