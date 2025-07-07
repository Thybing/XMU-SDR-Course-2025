import numpy as np
import matplotlib.pyplot as plt


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
def detect_preamble_cross_correlation(preamble, signal, plots=False, threshold=0.8):
    m_n = normalized_complex_xcorr(signal, preamble)

    if plots:
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
