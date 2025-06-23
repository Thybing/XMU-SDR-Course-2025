import math
import numpy as np
from matplotlib import pyplot

def complex_xcorr(arr, sub):
    arr = np.asarray(arr, dtype=np.complex128)
    sub = np.asarray(sub, dtype=np.complex128)
    N = len(arr)
    M = len(sub)
    ret_len = N - M + 1
    if ret_len <= 0:
        raise ValueError("sub is longer than arr")

    ret = np.zeros(ret_len, dtype=np.complex128)
    conj_sub = np.conj(sub)
    for i in range(ret_len):
        ret[i] = np.sum(arr[i:i+M] * conj_sub)
    return ret

def normalized_complex_xcorr(arr, sub):
    arr = np.asarray(arr, dtype=np.complex128)
    sub = np.asarray(sub, dtype=np.complex128)
    N = len(arr)
    M = len(sub)
    ret_len = N - M + 1
    if ret_len <= 0:
        raise ValueError("sub is longer than arr")

    conj_sub = np.conj(sub)
    sub_norm = np.sqrt(np.sum(np.abs(sub)**2))  # sub的范数，固定值

    ret = np.zeros(ret_len, dtype=np.complex128)
    arr_norm = np.zeros(ret_len, dtype=np.float64)

    for i in range(ret_len):
        arr_slice = arr[i:i+M]
        ret[i] = np.sum(arr_slice * conj_sub)
        arr_norm[i] = np.sqrt(np.sum(np.abs(arr_slice)**2))

    # 归一化，防止除以零
    denom = sub_norm * arr_norm
    denom[denom == 0] = 1e-15

    p_n = ret / denom
    return p_n

# Compare the correlation magnitude against this value to determine whether there is a preamble or not
def detect_preamble_cross_correlation(preamble, signal):
    m_n = normalized_complex_xcorr(signal, preamble)
    threshold = 0.9
    for i, val in enumerate(m_n):
        if val > threshold:
            return i
    return None


# This cell will test your implementation of `detect_preamble`
preamble_length = 100
signal_length = 1000
preamble = (np.random.random(preamble_length) + 1j *
            np.random.random(preamble_length))
signalA = np.random.random(signal_length) + 1j * np.random.random(signal_length)
signalB = np.random.random(signal_length) + 1j * np.random.random(signal_length)
preamble_start_idx = 123
signalB[preamble_start_idx:preamble_start_idx + preamble_length] += preamble
np.testing.assert_equal(detect_preamble_cross_correlation(preamble, signalA), None)
np.testing.assert_equal(detect_preamble_cross_correlation(preamble, signalB), preamble_start_idx)
