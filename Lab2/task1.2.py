import numpy as np
import matplotlib.pyplot as plt

preamble_lts = np.load("./Lab2_data/preamble_lts.npy")
preamble_sts = np.load("./Lab2_data/preamble_sts.npy")
rx_signal = np.load('Lab2_data/recorded_signal_strong.npy')  # 加载 .npy 文件

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16


def normalized_complex_xcorr(arr, sub):
    arr = np.asarray(arr, dtype=np.complex128)
    sub = np.asarray(sub, dtype=np.complex128)
    N = len(arr)
    M = len(sub)
    ret_len = N - M + 1
    if ret_len <= 0:
        raise ValueError("sub is longer than arr")

    conj_sub = np.conj(sub)
    sub_norm = np.sqrt(np.sum(np.abs(sub) ** 2))  # sub的范数，固定值

    ret = np.zeros(ret_len, dtype=np.complex128)
    arr_norm = np.zeros(ret_len, dtype=np.float64)

    for i in range(ret_len):
        arr_slice = arr[i:i + M]
        ret[i] = np.sqrt(np.abs(np.sum(arr_slice * conj_sub)) ** 2)
        arr_norm[i] = np.sqrt(np.sum(np.abs(arr_slice) ** 2))

    # 归一化，防止除以零
    denom = sub_norm * arr_norm
    denom[denom == 0] = 1e-15

    p_n = ret / denom
    return p_n


# Compare the correlation magnitude against this value to determine whether there is a preamble or not
def detect_preamble_cross_correlation(preamble, signal):
    m_n = normalized_complex_xcorr(signal, preamble)
    threshold = 0.78

    plt.figure(figsize=(10, 4))
    plt.plot(m_n.real, label='Cross-correlation coefficient')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('Cross-correlation Coefficient vs. Sample Index')
    plt.xlabel('Sample Index')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    starts = []
    resume_threshold = 300
    last_cnt = -resume_threshold

    for i, val in enumerate(m_n):
        if val > threshold and i >= last_cnt + resume_threshold:
            starts.append(i)
            last_cnt = i
    return starts


sts_starts = detect_preamble_cross_correlation(preamble_sts, rx_signal)
print(sts_starts)

for sts_start in sts_starts:
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

    frame_signal = rx_signal[sts_start: sts_start + 1920]
    # 4. CFO补偿
    time_vec = np.arange(len(frame_signal))  # 时间轴（采样点）
    correction = np.exp(-1j * 2 * np.pi * cfo_est * time_vec)
    frame_compensated = frame_signal * correction

    sts_signal = frame_compensated[: len_sts * 10]
    lts_signal = frame_compensated[len_sts * 10: + len_sts * 10 + int(len_lts * 2.5)]

    ofdm_data_symbols = frame_compensated[
                        len_sts * 10 + int(len_lts * 2.5):  len_sts * 10 + int(
                            len_lts * 2.5) + 80 * 20]

    # FFT
    n = 64
    cp_len = 16
    sym_len = n + cp_len
    num_symbol = 20

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

    H_est_debug = False
    if H_est_debug:
        # 子载波索引
        subcarriers = np.arange(len(H_est))

        # 幅度（模）
        magnitude = np.abs(H_est)

        # 相位（角度）
        phase = np.angle(H_est)

        # 绘图
        plt.figure(figsize=(12, 5))

        # 幅度响应
        plt.subplot(1, 2, 1)
        plt.stem(subcarriers, magnitude, use_line_collection=True)
        plt.title("Channel Magnitude Response")
        plt.xlabel("Subcarrier Index")
        plt.ylabel("|H[k]|")
        plt.grid(True)

        # 相位响应
        plt.subplot(1, 2, 2)
        plt.stem(subcarriers, phase, use_line_collection=True)
        plt.title("Channel Phase Response")
        plt.xlabel("Subcarrier Index")
        plt.ylabel("∠H[k] (radians)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

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

    # 导频子载波位置（标准802.11）
    pilot_indices = np.array([7, 21, 43, 57])

    # 已知导频值（BPSK）
    pilot_known = np.array([1, 1, 1, -1])

    # 5.2 估计每个符号的相位偏差（从导频）
    # shape: (4, num_symbols)
    pilot_received = equalized_symbols[pilot_indices, :]

    # 每个 pilot 的相位误差 = 角度差（received / known）
    # broadcasting: pilot_known[:, np.newaxis] 是 shape (4, 1)
    pilot_phase_errors = np.angle(pilot_received / pilot_known[:, np.newaxis])  # shape: (4, num_symbols)

    # 5.3 在频域上插值出 64 个子载波的相位误差
    # 初始化结果：相位偏差矩阵，shape (64, num_symbols)
    subcarrier_phase_offsets = np.zeros_like(equalized_symbols, dtype=float)

    for sym_idx in range(equalized_symbols.shape[1]):
        # 当前符号对应的导频相位
        phi = pilot_phase_errors[:, sym_idx]

        # 插值到 64 个子载波（线性插值）
        subcarrier_phase_offsets[:, sym_idx] = np.interp(
            np.arange(64),  # 所有子载波索引
            pilot_indices,  # 导频索引
            phi  # 导频相位
        )

    # 5.4 相位补偿
    # 生成复数补偿因子并作用于 equalized_symbols
    phase_correction = np.exp(-1j * subcarrier_phase_offsets)
    equalized_symbols = equalized_symbols * phase_correction

    # 2. 提取数据子载波
    data_symbols = equalized_symbols[data_subcarriers, :]  # (48, num_symbol)

    # 3. 拼接所有数据子载波（按时间序列排列）
    data_all = data_symbols.T.flatten()

    # 4. BPSK 解调
    demod_bits = (data_all.real > 0).astype(np.uint8)

    # 读取原始数据
    raw_data = np.load("./Lab2_data/raw_data.npy")

    length = min(len(raw_data), len(demod_bits))
    raw_data = raw_data[:length]
    demod_bits = demod_bits[:length]

    # 计算误码个数
    num_errors = np.sum(raw_data != demod_bits)

    # 计算误码率
    ber = num_errors / length

    print(f"比特长度: {length}")
    print(f"错误比特数: {num_errors}")
    print(f"比特误码率 (BER): {ber:.6f}")
