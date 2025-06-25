import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def normalized_shifted_cross_correlation(sig, L):
    N = len(sig)

    result_len = N - 2 * L + 1
    if result_len <= 0:
        raise ValueError("Signal too short for given L")

    sig = np.asarray(sig)
    part1 = sig[:N - L]
    part2 = sig[L:]

    # 滑动窗口内积（相关项）
    prod = part1 * np.conj(part2)
    numerator = np.convolve(prod, np.ones(L, dtype=prod.dtype), mode='valid')

    # 滑动能量（模长平方和）
    energy1 = np.abs(part1) ** 2
    energy2 = np.abs(part2) ** 2
    energy1_sum = np.convolve(energy1, np.ones(L), mode='valid')
    energy2_sum = np.convolve(energy2, np.ones(L), mode='valid')

    denominator = np.sqrt(energy1_sum * energy2_sum)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_n = np.abs(numerator) / denominator
        r_n[np.isnan(r_n)] = 0.0  # 避免除零出现 nan

    return r_n


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
