import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

import numpy as np


# def shifted_cross_correlation(sig, L):
#     sig = np.asarray(sig)
#     N = len(sig)
#     result_len = N - 2 * L + 1
#
#     if result_len <= 0:
#         raise ValueError("Signal too short for given L")
#
#     c = np.zeros(result_len, dtype=np.complex128)
#     for i in range(result_len):
#         x = sig[i:i + L]
#         y = sig[i + L:i + 2 * L]
#         c[i] = np.sum(x * np.conj(y))
#
#     return c
#
#
# def half_normalized_shifted_cross_correlation(sig, L):
#     sig = np.asarray(sig)
#     N = len(sig)
#     result_len = N - 2 * L + 1
#
#     if result_len <= 0:
#         raise ValueError("Signal too short for given L")
#
#     c = np.zeros(result_len, dtype=np.complex128)
#     for i in range(result_len):
#         x = sig[i:i + L]
#         y = sig[i + L:i + 2 * L]
#         numerator = np.sum(x * np.conj(y))
#         denominator = np.sum(np.abs(x) ** 2)
#         c[i] = np.abs(numerator) / denominator if denominator != 0 else 0.0
#
#     return c


def normalized_shifted_cross_correlation(sig, L):
    sig = np.asarray(sig)
    N = len(sig)
    result_len = N - 2 * L + 1

    if result_len <= 0:
        raise ValueError("Signal too short for given L")

    c = np.zeros(result_len, dtype=np.complex128)
    for i in range(result_len):
        x = sig[i:i + L]
        y = sig[i + L:i + 2 * L]
        numerator = np.sum(x * np.conj(y))
        denominator = np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
        c[i] = np.abs(numerator) / denominator if denominator != 0 else 0.0

    return c


def detect_preamble_auto_correlation(signal, short_preamble_len):
    c_n = normalized_shifted_cross_correlation(signal, short_preamble_len)
    threshold = 0.95

    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(c_n), label='|c_n| (normalized correlation)', color='blue')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Normalized Shifted Cross Correlation")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized |c_n|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for i, val in enumerate(c_n):
        if val > threshold:
            return i
    return None


# This cell will test your implementation of `detect_preamble`
short_preamble_length = 20
signal_length = 1000
short_preamble = np.exp(2j * np.pi * np.random.random(short_preamble_length))
preamble = np.tile(short_preamble, 10)  # 重复十次
noise = np.random.normal(size=signal_length) + 1j * np.random.normal(size=signal_length)
signalA = 0.1 * noise
signalB = 0.1 * noise
preamble_start_idx = 321
signalB[preamble_start_idx:preamble_start_idx + len(preamble)] += preamble
np.testing.assert_equal(detect_preamble_auto_correlation(signalA, short_preamble_length), None)
np.testing.assert_equal(
    detect_preamble_auto_correlation(signalB, short_preamble_length) in range(preamble_start_idx - 5,
                                                                              preamble_start_idx + 5), True)
