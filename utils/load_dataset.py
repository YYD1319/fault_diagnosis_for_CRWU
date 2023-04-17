import torch
import torch.utils.data as data
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
import numpy as np
from utils import preprocess


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


def load_2Ddata(cfg):
    def dataset_sampler(dataset, val_percentage=0.1):
        """
        split dataset into train set and val set
        :param dataset:
        :param val_percentage: validation percentage
        :return: split sampler
        """
        sample_num = len(dataset)
        file_idx = list(range(sample_num))
        train_idx, val_idx = train_test_split(file_idx, test_size=val_percentage, random_state=42)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        # https://blog.csdn.net/weixin_43914889/article/details/104607114
        # https://blog.csdn.net/qq_39355550/article/details/82688014
        return train_sampler, val_sampler

    def load_data(path, size, val_percentage, batch_size):
        data_transfrom = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize((size, size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(path, transform=data_transfrom)  # 没有transform，先看看取得的原始图像数据

        train_sampler, val_sampler = dataset_sampler(dataset, val_percentage)

        # dataloader定义
        train_iter = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=train_sampler)
        test_iter = DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=val_sampler)

        return train_iter, test_iter

    return load_data(cfg["path"], cfg["size"], cfg["val_percentage"], cfg["batch_size"])