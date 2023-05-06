from scipy.fftpack import fft, ifft, fftfreq, hilbert
import numpy as np
import matplotlib.pyplot as plt
from utils.signal_process.load_dataset import data_acquision
from time_domain_analysis import plt_raw_signal


def plt_FFT(data, sampling_period):
    # fft
    # fft_length = int(2 ** np.ceil(np.log2(data.size)))  # 将其设为2的幂次方，以保证FFT算法的效率和正确性
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    pos_Y_from_fft = Y[:Y.size // 2]

    freq = fftfreq(Y.size, d=sampling_period)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    # 绘制图像
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    axs.plot(freq, pos_Y_from_fft)
    axs.set_xlabel('Frequency (Hz)')
    axs.set_ylabel('Power Spectral Density')
    axs.set_title('FFT of Signal')
    axs.text(0.9, 0.9, 'Sampling Rate: {:.2f} Hz'.format(1 / sampling_period), ha='right', va='top', transform=axs.transAxes)
    axs.grid(True)  # 添加网格线

    plt.tight_layout()
    plt.show()


def plt_power_spectrum(data, sampling_period):
    # FFT
    # fft_length = int(2 ** np.ceil(np.log2(data.size)))
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    # power spectrum
    ps = Y ** 2 / Y.size
    ps = ps[:ps.size // 2]

    freq = fftfreq(Y.size, d=sampling_period)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制功率谱密度
    axs.plot(freq, ps)
    axs.set_title('Power Spectrum Density')
    axs.set_xlabel('Frequency (Hz)')
    axs.set_ylabel('Power')
    axs.grid(True)  # 添加网格线

    plt.show()


def plt_cepstrum(data, sampling_period):

    # fft_length = int(2 ** np.ceil(np.log2(data.size)))
    Y = fft(data)
    Y = np.abs(Y)
    Y = Y / Y.size * 2

    ceps = ifft(np.log(Y)).real
    ceps = ceps[:ceps.size // 2]
    ceps = np.abs(ceps)

    freq = fftfreq(Y.size, d=sampling_period)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size // 2)]  # 获取正频率

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制功率谱密度
    axs.plot(freq, ceps)
    axs.set_title('Cepstrum')
    axs.set_xlabel('Quefrency (s)')
    axs.set_ylabel('Magnitude')
    axs.grid(True)  # 添加网格线
    axs.set_ylim([0, 0.2])

    plt.tight_layout()
    plt.show()


def plt_envelope_spectrum(data, sampling_period, xlim=None, vline=None):
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

    freq = np.fft.fftfreq(at.size, d=sampling_period)  # 获取fft频率，此时包括正频率和负频率
    freq = freq[0:int(freq.size / 2)]  # 获取正频率

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    # 绘制功率谱密度
    axs.plot(freq, am)
    axs.set_title('Envelope Spectrum')
    axs.set_xlabel('freq(Hz)')
    axs.set_ylabel('amp(m/s2)')
    axs.grid(True)  # 添加网格线
    axs.set_ylim([0, 0.5])

    # if vline:  # 是否绘制垂直线
    #     plt.vlines(x=vline, ymax=0.2, ymin=0, colors='r')  # 高度y 0-0.2，颜色红色
    # if xlim:  # 图片横坐标是否设置xlim
    #     plt.xlim(0, xlim)
    plt.show()

if __name__ == "__main__":
    fs = 12000
    sampling_period = 1.0 / fs

    file_path = r'../../datasets/test/147.mat'
    data = data_acquision(file_path)

    plt_raw_signal(data)
    plt_FFT(data, sampling_period)
    plt_power_spectrum(data, sampling_period)
    plt_cepstrum(data, sampling_period)
    plt_envelope_spectrum(data=data, sampling_period=sampling_period, xlim=6000, vline=103)
