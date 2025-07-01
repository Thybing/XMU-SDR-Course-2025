import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

preamble_lts = np.load("./Lab1_data/preamble_lts.npy")
preamble_sts = np.load("./Lab1_data/preamble_sts.npy")
received_signal_strong = np.load("./Lab1_data/recorded_signal.npy")

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16


def calc_energy(sig):
    return np.abs(sig) ** 2


def estimate_noise_threshold(energy, head_length=1000, factor=5):
    noise_floor = np.mean(energy[:head_length])
    return noise_floor * factor


def detect_preamble_by_energy(signal):
    energy = calc_energy(signal)
    energy_threshold = estimate_noise_threshold(energy, 1000, 10)
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

    # |——10STS——|—2.5LTS—|———20 OFDM data symbols———|
    data_starts = []
    cur = 0

    idle = True
    resume_cnt = 0
    resume_threshold = 200
    while cur < len(energy) - 1:
        if energy[cur] > energy_threshold:
            resume_cnt = 0
            if idle:
                data_starts.append(cur + 1 + 10 * len_sts + int(2.5 * len_lts))
                idle = False
        elif not idle:
            resume_cnt += 1
            if (resume_cnt >= resume_threshold):
                idle = True

        cur += 1

    return data_starts


def estimate_snr(received_signal, signal_slice, noise_slice):
    signal_power = np.mean(np.abs(received_signal[signal_slice]) ** 2)
    noise_power = np.mean(np.abs(received_signal[noise_slice]) ** 2)
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


data_starts = detect_preamble_by_energy(received_signal_strong)
print("strong signal data_starts:", end='')
print(data_starts)
print("strong signal psnr {}".format(
    estimate_snr(received_signal_strong, slice(data_starts[0] - 320, data_starts[0]), slice(30))))
