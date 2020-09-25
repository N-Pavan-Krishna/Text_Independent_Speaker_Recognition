import numpy
import scipy.io.wavfile
import scipy.fftpack
import glob

import os


# I used following python packages:
# numpy
# scipy
# reference : Mel Frequency cepstral Coefficients (mfcc) tutorial, practical cryptography
#             stackoverflow
#             python documentation

def pre_emphasis(data):
    # Objective: To amplify the importance of high-frequency in the signal
    # input : data=takes signal as input
    # output: emphasised_data = amplified signal
    alpha = 0.97  # filter coefficient
    emphasised_data = numpy.append(data[0], data[1:] - alpha * data[:-1])
    return emphasised_data


def framing(emphasized_data, sample_rate):
    # Objective : Frequency in a signal changes over time, to avoid that we split signal into short frames
    # input : emphasized data= takes emphasised signal
    #        sample_rate = takes sample rate of the signal
    # output: frames and frame length
    size = 0.025  # frame size
    step = 0.01  # frame step

    frame_len, frame_step = size * sample_rate, step * sample_rate
    data_length = len(emphasized_data)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(data_length - frame_len)) / frame_step))

    # we usually want frame size to be in powers of 2 to facilitate the use of FFT
    # so we do padding of zeros to the nearest power of 2
    pad_data_length = num_frames * frame_step + frame_len
    pad = numpy.zeros((pad_data_length - data_length))
    pad_data = numpy.append(emphasized_data, pad)

    indices = numpy.tile(numpy.arange(0, frame_len), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_data[indices.astype(numpy.int32, copy=False)]

    return frames, frame_len


def window(frames, frame_len):
    # Objective: To reduce spectral leakage by applying hamming code
    # input: frames and frame_length
    # output: frames with hamming code
    window_hamming = numpy.hamming(frame_len)
    frames = frames * window_hamming
    return frames


def fourier_transform(frames):
    nfft = 512
    magnitude_frames = numpy.absolute(numpy.fft.rfft(frames, nfft))
    power_frames = ((1.0 / nfft) * (magnitude_frames ** 2))
    return power_frames


def filter(sample_rate, power_frames):
    # input: fourier transformed frames and sample rate of signal
    # output: filter bank which are linear
    # we calculate he mel frequency around frames
    num_filters = 40  # number of filters
    nftt = 512  # filter size

    mel_low_frequency = 0
    mel_high_frequency = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(mel_low_frequency, mel_high_frequency, num_filters + 2)
    horizantal_points = (700 * (10 ** (mel_points / 2595) - 1))
    value = numpy.floor((nftt + 1) * horizantal_points / sample_rate)

    fbank = numpy.zeros((num_filters, int(numpy.floor(nftt / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = int(value[m - 1])
        f_m = int(value[m])
        f_m_plus = int(value[m + 1])

        for i in range(f_m_minus, f_m):
            fbank[m - 1, i] = (i - value[m - 1]) / (value[m] - value[m - 1])
        for j in range(f_m, f_m_plus):
            fbank[m - 1, j] = (value[m + 1] - j) / (value[m + 1] - value[m])
    fil_bank = numpy.dot(power_frames, fbank.T)
    fil_bank = numpy.where(fil_bank == 0, numpy.finfo(float).eps, fil_bank)
    fil_bank = 20 * numpy.log10(fil_bank)

    return fil_bank


def lift(mfcc):
    # we apply lift to enhance the coefficients of MFCC
    ceptral_lift = 22
    num_frames, num_coeff = mfcc.shape
    n = numpy.arange(num_coeff)
    lifter = 1 + (ceptral_lift / 2) * numpy.sin(numpy.pi * n / ceptral_lift)
    return mfcc * lifter


def MFCC(filename):
    # This function take audio file as input and produces the feature vector as output
    sample_rate, data = scipy.io.wavfile.read(filename)
    emphasized_data = pre_emphasis(data)
    # print(signal_data.shape)
    frames, frame_len = framing(emphasized_data, sample_rate)
    # print(frames.shape)
    power_frames = fourier_transform(frames)
    frames = window(frames, frame_len)
    fil_bank = filter(sample_rate, power_frames)
    num_cep_coefficients = 12
    mfcc = scipy.fftpack.dct(fil_bank, type=2, axis=1, norm='ortho')[:, 1: (num_cep_coefficients + 1)]
    mfcc = lift(mfcc)

    return mfcc
