#coding:utf-8
"""
Edited on Fri Mar 22 10:20:32 2019

@author: zhang
"""
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
# import numpy as np
# np.set_printoptions(threshold=np.inf)  # 设置打印选项：输出数组元素数目上限为无穷

def prepro(d_path, length=864, number=1000, normal=True, rate=[0.7, 0.2, 0.1], enc=True, enc_step=28):
    """对数据进行预处理(随机采样),返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.

    :param d_path: 源数据地址
    :param length: 信号长度.
    :param number: 每种信号个数. 每类number个样本（共10类，每个样本长length）
    :param normal: 是否标准化.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    """
    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典

        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        for i in filenames:  # 遍历每一个文件的DE数据
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:  # 驱动端振动数据
                    # print(key)
                    files[i] = file[key].ravel()

        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.

        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))  # 训练集尾下标
            samp_train = int(number * (1 - slice_rate))  # 训练集样本数
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step  # 每次采集的训练样本长length(864)，偏移量为enc_step(28)，则数据增强前每次采集的样本，经过数据增强后，可以产生30个前后有重叠的样本 864//28
                # ((N-L)/S)+1
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))  # 如果只- length，则最后一次采集样本无法数据增强
                    label = 0
                    for h in range(enc_time):  # 对每次采集的样本进行数据增强
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:  # 训练样本仍然是700
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    # random_start = np.random.randint(low=0, high=(end_index - length))
                    random_start = np.random.randint(low=0, high=(all_lenght - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)
            # 抓取测试数据
            # for h in range(number - samp_train):
            #     random_start = np.random.randint(low=end_index, high=(all_lenght - length))
            #     sample = slice_data[random_start:random_start + length]

            for h in range(len(slice_data) // length):
                random_start = np.random.randint(low=0, high=(len(slice_data) - length))
                # 每次取j至j+length长度的数据
                sample = slice_data[h * length:(h + 1) * length]
                # sample = slice_data[random_start:random_start + length]

                Test_Sample.append(sample)

            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:  # 遍历每一个文件的文件名，key
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)  # 确定编码的对应关系
        Train_Y = Encoder.transform(Train_Y).toarray()  # 编码
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])  # 将0.3的测试集，划分为0.2：0.1的验证集：测试集
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)  # 保证划分后的每个样本中，类别比例和划分前相同
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    def show_config(train_shape, valid_shape, test_shape):
        print('训练样本维度:', train_shape)
        print(train_shape[0], '训练样本个数')
        print('验证样本的维度', valid_shape)
        print(valid_shape[0], '验证样本个数')
        print('测试样本的维度', test_shape)
        print(test_shape[0], '测试样本个数')

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)

    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    # Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    Train_Y, Test_Y = np.asarray(Train_Y, dtype=np.int64), np.asarray(Test_Y, dtype=np.int64)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    show_config(Train_X.shape, Valid_X.shape, Test_X.shape)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    path = "../datasets/48k_DE_data/0HP"
    #path = '/home/zjl/wdcnn_bearning_fault_diagnosis/data/0HP'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=True,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=True,
                                                                enc_step=28)
    # print(train_X)