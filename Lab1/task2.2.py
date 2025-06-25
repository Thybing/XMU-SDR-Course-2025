import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy.matlib import zeros

preamble_lts = np.load("./Lab1_data/preamble_lts.npy")
preamble_sts = np.load("./Lab1_data/preamble_sts.npy")
received_signal_weak = np.load("./Lab1_data/recorded_signal_weak.npy")
received_signal_strong = np.load("./Lab1_data/recorded_signal_strong.npy")

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16


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

    threshold = 5

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

    starts = []
    while 1:
        max_value = np.max(m)
        max_index = np.argmax(m)

        if max_value < threshold:
            starts.sort()
            return starts
        starts.append(max_index + short_preamble_len)
        m[max_index - short_preamble_len: max_index + short_preamble_len] = 0


sts_starts = detect_preamble_by_sliding_window(received_signal_strong, len_sts)
sts_starts = np.array(sts_starts)
data_starts = sts_starts + (10 * len_sts + int(2.5 * len_lts))
print("strong signal data_starts:", end='')
print(data_starts)

sts_starts = detect_preamble_by_sliding_window(received_signal_weak, len_sts)
sts_starts = np.array(sts_starts)
data_starts = sts_starts + (10 * len_sts + int(2.5 * len_lts))
print("weak signal data_starts:", end='')
print(data_starts)
