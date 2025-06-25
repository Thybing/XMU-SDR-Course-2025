import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def calc_energy(sig):
    return np.abs(sig) ** 2


def detect_preamble_by_sliding_window(signal, short_preamble_len):
    energy = calc_energy(signal)
    length = len(signal) - short_preamble_len + 1
    if length - short_preamble_len <= 0:
        print("signal too short")
        return None
    energy_window_sum = np.zeros(length)
    energy_window_sum[0] = sum(energy[0:short_preamble_len])
    for i in range(1, length):
        energy_window_sum[i] = energy_window_sum[i - 1] - energy[i - 1] + energy[i + short_preamble_len - 1]

    pa = energy_window_sum[0:length - short_preamble_len]
    pb = energy_window_sum[short_preamble_len:length]
    m = pb / pa
    max_value = np.max(m)
    max_index = np.argmax(m)

    threshold = 30

    # 画图
    plt.figure(figsize=(10, 4))
    plt.plot(m, label=f'Energy Window Sum (window size={short_preamble_len})')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Sample index')
    plt.ylabel('Energy sum')
    plt.title('Sliding Window Energy Sum')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if max_value > threshold:
        return max_index + short_preamble_len

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
np.testing.assert_equal(detect_preamble_by_sliding_window(signalA, short_preamble_length), None)
np.testing.assert_equal(
    detect_preamble_by_sliding_window(signalB, short_preamble_length) in range(preamble_start_idx - 5,
                                                                               preamble_start_idx + 5), True)
