import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

preamble_lts = np.load("./Lab1_data/preamble_lts.npy")
preamble_sts = np.load("./Lab1_data/preamble_sts.npy")
received_signal_weak = np.load("./Lab1_data/recorded_signal_weak.npy")
received_signal_strong = np.load("./Lab1_data/recorded_signal_strong.npy")

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16


def half_normalized_shifted_cross_correlation(sig, L):
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
        denominator = np.sum(np.abs(x) ** 2)
        c[i] = np.abs(numerator) / denominator if denominator != 0 else 0.0

    return c


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


def detect_preamble_auto_correlation(signal, short_preamble_len, threshold):
    c_n = normalized_shifted_cross_correlation(signal, short_preamble_len)

    plt.figure(figsize=(10, 4))
    plt.plot(c_n.real, label='|c_n| (normalized correlation)', color='blue')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Normalized Shifted Cross Correlation")
    plt.xlabel("Sample index")
    plt.ylabel("Normalized |c_n|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    starts = []
    idle = True
    resume_cnt = 0
    for i, val in enumerate(c_n):
        if idle and val > threshold:
            starts.append(i)
            idle = False
            resume_cnt = 0
        elif not idle:
            resume_cnt += 1
            if resume_cnt > 300:
                idle = True

    return starts


starts = np.array(detect_preamble_auto_correlation(received_signal_strong, len_sts, 0.998))
starts += len_sts * 10 + int(2.5 * len_lts)
print("strong signal data starts calculated by preamble_sts:", end='')
print(starts)

starts = np.array(detect_preamble_auto_correlation(received_signal_weak, len_sts, 0.8))
starts += len_sts * 10 + int(2.5 * len_lts)
print("weak signal data starts calculated by preamble_sts:", end='')
print(starts)
