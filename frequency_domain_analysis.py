from scipy.io import loadmat
from scipy.fftpack import fft, ifft, fftfreq, hilbert
import numpy as np
import matplotlib.pyplot as plt

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


def plt_FFT(data):
    # fft
    # fft_length = int(2 ** np.ceil(np.log2(data.size)))  # 将其设为2的幂次方，以保证FFT算法的效率和正确性
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    pos_Y_from_fft = Y[:Y.size // 2]

    freq = fftfreq(Y.size, d=t_s)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    plt.figure(figsize=(20, 12))

    plt.subplot(311)
    plt.plot(data)

    plt.subplot(212)
    # plt.plot(freq, pos_Y_from_fft)
    plt.plot(pos_Y_from_fft)
    plt.show()


def plt_power_spectrum(data):
    # FFT
    # fft_length = int(2 ** np.ceil(np.log2(data.size)))
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    # power spectrum
    ps = Y ** 2 / Y.size
    ps = ps[:ps.size // 2]

    freq = fftfreq(Y.size, d=t_s)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    plt.figure(figsize=(30, 12))

    plt.subplot(211)
    plt.plot(data)

    plt.subplot(212)
    # plt.plot(20 * np.log10(ps[:fft_length // 2]))
    # plt.plot(freq, ps)
    plt.plot(ps)
    plt.show()


def plt_cepstrum(data):

    # fft_length = int(2 ** np.ceil(np.log2(data.size)))
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    ceps = ifft(np.log(Y)).real
    ceps = ceps[:ceps.size // 2]
    ceps = np.abs(ceps)

    freq = fftfreq(Y.size, d=t_s)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    plt.figure(figsize=(30, 12))

    plt.subplot(211)
    plt.plot(data)

    plt.subplot(212)
    # plt.plot(freq, ceps)
    plt.plot(ceps)
    plt.ylim([0, 0.2])
    plt.show()


def plt_envelope_spectrum(data, fs, xlim=None, vline=None):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None
    param vline: 图片垂直线，default = None
    '''
    # ----去直流分量----#
    # data = data - np.mean(data)
    # ----做希尔伯特变换----#
    xt = data
    ht = hilbert(xt)
    at = np.sqrt(xt ** 2 + ht ** 2)  # 获得解析信号at = sqrt(xt^2 + ht^2)
    # ----做fft变换----#
    am = fft(at)  # 对解析信号at做fft变换获得幅值
    am = np.abs(am)  # 对幅值求绝对值（此时的绝对值很大）
    am = am / am.size * 2
    am = am[0: int(am.size / 2)]  # 取正频率幅值

    freq = np.fft.fftfreq(at.size, d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size / 2)]  # 获取正频率
    plt.plot(freq, am)

    # if vline:  # 是否绘制垂直线
    #     plt.vlines(x=vline, ymax=0.2, ymin=0, colors='r')  # 高度y 0-0.2，颜色红色
    # if xlim:  # 图片横坐标是否设置xlim
    #     plt.xlim(0, xlim)
    plt.xlabel('freq(Hz)')  # 横坐标标签
    plt.ylabel('amp(m/s2)')  # 纵坐标标签
    plt.show()

if __name__ == "__main__":
    t_s = 0.01
    t_start = 0.5
    t_end = 5
    t = np.arange(t_start, t_end, t_s)

    f0 = 5
    f1 = 20

    # generate the orignal signal
    y = 1.5 * np.sin(2 * np.pi * f0 * t) + 3 * np.sin(2 * np.pi * 20 * t) + np.random.randn(t.size)

    file_path = r'./147.mat'
    data = data_acquision(file_path)
    plt_envelope_spectrum(data=data, fs=12000, xlim=6000, vline=103)

    plt_FFT(data)
    plt_power_spectrum(data)
    plt_cepstrum(data)
