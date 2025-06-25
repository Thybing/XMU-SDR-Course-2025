import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def normalized_shifted_cross_correlation(sig, L):
    N = len(sig)
    result_len = N - 2 * L + 1
    if result_len <= 0:
        raise ValueError("Signal too short for given L")

    r_n = np.zeros(result_len)
    for i in range(result_len):
        part1 = sig[i: i + L]
        part2 = sig[i + L: i + 2 * L]
        numerator = np.abs(np.sum(part1 * np.conj(part2)))

        energy1 = np.sum(np.abs(part1) ** 2)
        energy2 = np.sum(np.abs(part2) ** 2)

        denominator = np.sqrt(energy1 * energy2)

        if denominator != 0:
            r_n[i] = numerator / denominator
        else:
            r_n[i] = 0.0

    return r_n


def detect_preamble_auto_correlation(signal, short_preamble_len):
    c_n = normalized_shifted_cross_correlation(signal, short_preamble_len)

    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(c_n), label='|c_n| (normalized correlation)', color='blue')
    plt.title("Normalized Shifted Cross Correlation")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized |c_n|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    threshold = 0.95
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
# np.testing.assert_equal(detect_preamble_auto_correlation(signalA, short_preamble_length), None)
np.testing.assert_equal(
    detect_preamble_auto_correlation(signalB, short_preamble_length) in range(preamble_start_idx - 5,
                                                                              preamble_start_idx + 5), True)
