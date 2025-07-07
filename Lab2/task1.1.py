import numpy as np
import matplotlib.pyplot as plt

preamble_lts = np.load("./Lab2_data/preamble_lts.npy")
preamble_sts = np.load("./Lab2_data/preamble_sts.npy")
tx_signal = np.load('Lab2_data/tx_signal.npy')  # 加载 .npy 文件

len_lts = len(preamble_lts)  # 64
len_sts = len(preamble_sts)  # 16

# ofdm_data提取

ofdm_data_symbols = tx_signal[320:]

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

n = 64
lts_len = n * 2
lts_start = 160 + 32
lts_end = lts_start + lts_len

rx_lts = tx_signal[lts_start:lts_end]  # 长度 128
lts1 = rx_lts[:n]
lts2 = rx_lts[n:]

ideal_lts = preamble_lts[:n]

rx_lts1_freq = np.fft.fft(lts1, n)
rx_lts2_freq = np.fft.fft(lts2, n)
ideal_lts_freq = np.fft.fft(ideal_lts, n)

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
