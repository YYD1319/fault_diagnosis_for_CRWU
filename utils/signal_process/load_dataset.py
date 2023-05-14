import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
from utils.signal_process import preprocess


def data_acquision(file_path):
    """
    fun: 从cwru mat文件读取加速度数据
    param file_path: mat文件绝对路径
    return accl_data: 加速度数据，array类型
    """
    file = loadmat(file_path)  # 加载mat数据
    # data_key_list = list(data.keys())  # mat文件为字典类型，获取字典所有的键并转换为list类型
    file_keys = file.keys()
    for key in file_keys:
        if 'DE' in key:  # 驱动端振动数据
            data = file[key].ravel()
    # accl_key = data_key_list[3]  # 获取'X108_DE_time'
    # accl_data = data[accl_key].flatten()  # 获取'X108_DE_time'所对应的值，即为振动加速度信号,并将二维数组展成一维数组
    return data

def load_1Ddata(cfg):
    def load_array(data_arrays, batch_size, is_train=True):
        """构造一个PyTorch数据迭代器"""
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=cfg["path"],
                                                                           length=cfg["input_size"],
                                                                           number=cfg["number"],
                                                                           normal=cfg["normal"],
                                                                           rate=cfg["rate"],
                                                                           enc=cfg["enc"],
                                                                           enc_step=cfg["enc_step"])

    train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]
    # 添加维度 (7000, 864) -> (7000, 1, 864) [batchsize, input_size, max_length] [批量大小N， 序列长度L, 特征长度Hin]

    x_train, y_train, x_valid, y_valid, x_test, y_test = torch.tensor(train_X, dtype=torch.float), \
                                                         torch.tensor(train_Y, dtype=torch.long), \
                                                         torch.tensor(valid_X, dtype=torch.float), \
                                                         torch.tensor(valid_Y, dtype=torch.long), \
                                                         torch.tensor(test_X, dtype=torch.float), \
                                                         torch.tensor(test_Y, dtype=torch.long)
    train_iter = load_array((x_train, y_train), cfg["batch_size"])
    test_iter = load_array((x_test, y_test), cfg["batch_size"])
    valid_iter = load_array((x_valid, y_valid), cfg["batch_size"])

    return train_iter, test_iter, valid_iter

def get_kfold_dataset(dataset_path, rate, size):
    data_transfrom = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((size, size)),
        # transforms.RandomHorizontalFlip(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=data_transfrom)
    train_valid_size = int(rate * len(dataset))
    test_size = len(dataset) - train_valid_size
    train_valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                      [train_valid_size, test_size])

    return train_valid_dataset, test_dataset

def get_kfold_test_iter(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

def get_kfold_train_vaalid_iter(dataset, k, batch_size, shuffle=True):
    # 划分 k 个子集
    subsets = []
    subset_size = len(dataset) // k
    for i in range(k):
        start_index = i * subset_size
        end_index = start_index + subset_size
        if i == k - 1:
            end_index = len(dataset)
        subset = Subset(dataset, range(start_index, end_index))
        subsets.append(subset)

    # 生成 k 折交叉验证的训练集和验证集 DataLoader
    for i in range(k):
        val_subset = subsets[i]
        train_subsets = subsets[:i] + subsets[i + 1:]
        train_dataset = ConcatDataset(train_subsets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=shuffle)
        yield train_loader, val_loader, i



def load_2Ddata(dataset_path, rate, batch_size, size):
    data_transfrom = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((size, size)),
        # transforms.RandomHorizontalFlip(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=data_transfrom)

    train_size = int(rate[0] * len(dataset))
    valid_size = int(rate[1] * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                               [train_size, valid_size, test_size])

    # dataloader定义
    train_iter = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    return train_iter, valid_iter, test_iter
