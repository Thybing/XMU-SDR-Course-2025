import numpy as np
import matplotlib.pyplot as plt
import detect_preamble
from Lab2.detect_preamble import detect_preamble_cross_correlation

preamble_lts = np.load("./Lab2_data/preamble_lts.npy")
preamble_sts = np.load("./Lab2_data/preamble_sts.npy")
rx_signal = np.load('Lab2_data/recorded_signal_10sym.npy')  # 加载 .npy 文件

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16

sts_starts = detect_preamble.detect_preamble_cross_correlation(preamble_sts, rx_signal)
print("data_pack_starts:", end='')
print(sts_starts)

data_pack_index = 0
for sts_start in sts_starts:
    print(f"find ofdm data pack : {data_pack_index}")
    data_pack_index += 1

    num_symbol = 10

    lts_signal = rx_signal[sts_start + len_sts * 10: sts_start + len_sts * 10 + int(len_lts * 2.5)]

    # CFO补偿
    num_sts = 10
    sts_len = 16
    lts_segment_len = 64
    lts_len_total = 160  # 实际LTS段长度（含GI、CP等）

    lts1 = lts_signal[32: 32 + len_lts]
    lts2 = lts_signal[32 + len_lts: 32 + len_lts * 2]

    # 3. CFO估计：基于两段LTS的相位差
    product = np.sum(lts1 * np.conj(lts2))
    angle_diff = np.angle(product)
    cfo_est = angle_diff / (2 * np.pi * lts_segment_len)  # 单位是 归一化频偏（弧度/采样点）

    frame_signal = rx_signal[sts_start: sts_start + 320 + 80 * num_symbol]
    # 4. CFO补偿
    time_vec = np.arange(len(frame_signal))  # 时间轴（采样点）
    correction = np.exp(-1j * 2 * np.pi * cfo_est * time_vec)
    frame_compensated = frame_signal * correction

    sts_signal = frame_compensated[: len_sts * 10]
    lts_signal = frame_compensated[len_sts * 10: + len_sts * 10 + int(len_lts * 2.5)]

    ofdm_data_symbols = frame_compensated[
                        len_sts * 10 + int(len_lts * 2.5):  len_sts * 10 + int(
                            len_lts * 2.5) + 80 * num_symbol]

    # FFT
    n = 64
    cp_len = 16
    sym_len = n + cp_len

    freq_domain_symbols = []
    for i in range(num_symbol):
        start = i * sym_len + cp_len
        end = start + n
        time_domain_symbol = ofdm_data_symbols[start:end]
        freq_symbol = np.fft.fft(time_domain_symbol, n)
        freq_domain_symbols.append(freq_symbol)

    freq_domain_symbols = np.array(freq_domain_symbols).T  # shape: (64, 20)

    # 使用lts进行信道估计,计算出H

    lts1 = lts_signal[32: 32 + len_lts]
    lts2 = lts_signal[32 + len_lts: 32 + len_lts * 2]

    ideal_lts = preamble_lts[:len_lts]

    rx_lts1_freq = np.fft.fft(lts1, len_lts)
    rx_lts2_freq = np.fft.fft(lts2, len_lts)
    ideal_lts_freq = np.fft.fft(ideal_lts, len_lts)
    # H_est 是之前你计算得到的信道频率响应 (长度 64 的复数数组)

    # 平均两个接收 LTS，抗噪声
    rx_lts_avg_freq = (rx_lts1_freq + rx_lts2_freq) / 2

    # 避免除以 0：仅对有效子载波做估计
    H_est = np.zeros(n, dtype=complex)
    valid = np.abs(ideal_lts_freq) > 1e-6

    H_est[valid] = rx_lts_avg_freq[valid] / ideal_lts_freq[valid]

    # 均衡后的频域符号
    equalized_symbols = np.zeros_like(freq_domain_symbols, dtype=complex)
    valid = np.abs(H_est) > 1e-6  # 避免除以0
    equalized_symbols[valid, :] = freq_domain_symbols[valid, :] / H_est[valid, np.newaxis]

    n = 64  # FFT 点数
    num_symbol = equalized_symbols.shape[1]  # OFDM 符号数

    # 子载波索引（基于你给的频率映射）
    data_subcarriers = np.array(
        list(range(1, 7)) +
        list(range(8, 21)) +
        list(range(22, 27)) +
        list(range(38, 43)) +
        list(range(44, 57)) +
        list(range(58, 64))
    )
    pilot_subcarriers = np.array([7, 21, 43, 57])
    idle_subcarriers = np.array([0] + list(range(27, 38)))  # DC+空闲

    # 1. 信道均衡
    valid = np.abs(H_est) > 1e-6
    equalized_symbols = np.zeros_like(freq_domain_symbols, dtype=complex)
    equalized_symbols[valid, :] = freq_domain_symbols[valid, :] / H_est[valid, np.newaxis]

    # 7. 提取所有数据子载波，flatten
    data_symbols = equalized_symbols[data_subcarriers, :]
    data_all = data_symbols.T.flatten()

    if data_pack_index == 1:
        plt.figure(figsize=(6, 6))
        plt.plot(data_all.real, data_all.imag, 'bo', markersize=2)
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.title("Constellation of Data Subcarriers (After Equalization + Phase Correction)")
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.grid(True)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    # 4. BPSK 解调
    demod_bits = (data_all.real > 0).astype(np.uint8)

    length = len(demod_bits)

    print(f"比特长度: {length}")
    print(demod_bits[:20])
