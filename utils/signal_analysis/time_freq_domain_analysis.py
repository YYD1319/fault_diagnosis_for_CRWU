from scipy.io import loadmat
from utils.signal_process.wavelets_trans import cwt_trans
import numpy as np
import pywt
def data_acquision(file_path):
    """
    fun: 从cwru mat文件读取加速度数据
    param file_path: mat文件绝对路径
    return accl_data: 加速度数据，array类型
    """
    data = loadmat(file_path)  # 加载mat数据
    data_key_list = list(data.keys())  # mat文件为字典类型，获取字典所有的键并转换为list类型
    accl_key = data_key_list[3]  # 获取'X108_DE_time'
    accl_data = data[accl_key].flatten()  # 获取'X108_DE_time'所对应的值，即为振动加速度信号,并将二维数组展成一维数组
    return accl_data

if __name__ == "__main__":
    file_path = r'../../datasets/test/147.mat'
    data = data_acquision(file_path)

    N = data.size  # 和样本采样长度相同(length=784)
    fs = 12000  # 采样频率，和实际的采样频率相同
    t = np.linspace(0, N / fs, N, endpoint=False)
    wave_name = 'cmor3-3'
    total_scal = 256
    fc = pywt.central_frequency(wave_name)
    cparam = 2 * fc * total_scal  # 常数(n)影响图像分布区域
    scales = cparam / np.arange(total_scal, 1, -1)
    sampling_period = 1.0 / fs

    fig = cwt_trans(data, t, scales, wave_name, sampling_period)
    fig.show()