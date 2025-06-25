import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def calc_energy(sig):
    return np.abs(sig) ** 2


def detect_preamble_by_energy(signal):
    energy = calc_energy(signal)
    energy_threshold = 1

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(energy, color='purple', label='|sig[i]|² (energy)')
    plt.axhline(y=energy_threshold, color='r', linestyle='--', label='Threshold')
    plt.title("Signal Energy per Sample")
    plt.xlabel("Sample index")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for i, val in enumerate(energy):
        if val > energy_threshold:
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
np.testing.assert_equal(detect_preamble_by_energy(signalA), None)
np.testing.assert_equal(
    detect_preamble_by_energy(signalB) in range(preamble_start_idx - 5,
                                                preamble_start_idx + 5), True)
