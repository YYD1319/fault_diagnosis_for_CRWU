import numpy as np
import scipy.stats as stats
from utils.signal_process.load_dataset import data_acquision
import matplotlib.pyplot as plt

def plt_raw_signal(data):
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))

    # 绘制原始信号
    axs.plot(data)
    axs.set_title('Raw Signal')
    axs.set_xlabel('Time')
    axs.set_ylabel('Amplitude')

    plt.show()
def time_domain_analysis(data):
    # 基本统计指标
    max_value = np.max(data)
    min_value = np.min(data)
    peak_value = max_value - min_value  # 峰峰值
    mean_value = np.mean(data)
    abs_mean_value = np.mean(np.abs(data))  # 平均幅值
    sra_value = np.square(np.mean(np.sqrt(np.abs(data))))  # 方根幅值
    rms_value = np.sqrt(np.mean(np.square(data)))  # 均方根
    variance_value = np.var(data)  # 方差

    # 峭度和偏度
    kurtosis_value = stats.kurtosis(data)  # 数据分布的平坦度和尖度程度 3
    skewness_value = stats.skew(data)  # 数据分布的对称性 0

    # 峰值因子 (Crest factor)
    crest_factor = max_value / rms_value  # 反映随机变量分布特性的数值统计量 3

    # 波形因子 (Form factor)
    form_factor = rms_value / abs_mean_value  # 波形因子越接近于1，表示信号波形越接近于正弦波；波形因子大于1，表示信号波形与正弦波的形状差异较大

    # 脉冲因子 (Impulse factor)
    impulse_factor = max_value / abs_mean_value  # 反映了信号中极端值与其有效值（或均方根值，RMS）之比

    # 裕度因子 (Margin factor)
    margin_factor = max_value / sra_value

    # 返回结果
    return {
        "最大值": max_value,
        "最小值": min_value,
        "峰值": max_value,
        "峰峰值": peak_value,
        "均值": mean_value,
        "平均幅值": abs_mean_value,
        "均方根": rms_value,
        "方根幅值": sra_value,
        "方差": variance_value,
        "峭度": kurtosis_value,
        "偏度": skewness_value,
        "峭度指标 (Crest factor)": crest_factor,
        "波形因子 (Form factor)": form_factor,
        "峰值因子 (Impulse factor)": impulse_factor,
        "脉冲因子 (Margin factor)": margin_factor
    }


if __name__ == "__main__":
    file_path = r'../../datasets/test/147.mat'
    data = data_acquision(file_path)

    # 调用函数并输出结果
    result = time_domain_analysis(data)
    for key, value in result.items():
        print(key, ":", value)
