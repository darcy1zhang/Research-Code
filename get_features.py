import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, find_peaks
from scipy.interpolate import interp1d

def update_array(a, data_tmp):
    i = 0
    while i < len(a) - 2:
        if data_tmp[a[i]] < data_tmp[a[i + 1]] < data_tmp[a[i + 2]]:
            a = np.delete(a, i)
        elif data_tmp[a[i]] > data_tmp[a[i + 1]] > data_tmp[a[i + 2]]:
            a = np.delete(a, i + 2)
        else:
            i += 1
    return a

def delete_unique(a):
    mean = np.mean(a)
    std = np.std(a)
    # 设置阈值
    threshold = 1
    # 使用布尔索引删除特殊值
    filtered_a = a[np.abs(a - mean) <= threshold * std]
    return filtered_a

def get_features(data_tmp):
    max = np.max(data_tmp)
    min = np.min(data_tmp)
    data_tmp = data_tmp / max

    fs = 100
    t = np.linspace(0, 10, 10 * fs)
    signal = data_tmp

    # 峰值检测
    peak_indices, _ = find_peaks(signal)  # 返回极大值点的索引

    # 线性插值
    t_peaks = t[peak_indices]  # 极大值点的时间
    peak_values = signal[peak_indices]  # 极大值点的幅值
    interpolation_func = interp1d(t_peaks, peak_values, kind='linear', bounds_error=False, fill_value=0)
    envelope = interpolation_func(t)

    peaks2, _ = find_peaks(envelope, distance=10)

    peaks2 = update_array(peaks2, data_tmp)
    if len(peaks2) % 2 != 0:
        peaks2 = np.delete(peaks2, len(peaks2) - 1)

    diff_12 = peaks2[1::2] - peaks2[0::2]
    diff_21 = peaks2[2::2] - peaks2[1:-1:2]
    diff_22 = peaks2[2::2] - peaks2[:-2:2]

    m_12 = np.mean(diff_12)
    m_21 = np.mean(diff_21)
    m_22 = np.mean(diff_22)
    features = np.arrary([m_12, m_21, m_22])

    return features
    # print(diff_12)
    # print(diff_21)
    # print(diff_22)

data_test = np.load("data_test.npy")
data_tmp = data_test[4235, :1000]