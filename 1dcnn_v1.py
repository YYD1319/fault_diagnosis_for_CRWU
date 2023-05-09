# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.optim as optim
from utils.train import train
from utils.train import analyze
from utils.signal_process import load_dataset


class CNN1D(nn.Module):
    def __init__(self, dropout):
        super(CNN1D, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=20, stride=8, padding=598),  # torch.Size([128, 32, 256]) [N, C_out, L_out]
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # torch.Size([128, 32, 64])
            nn.Conv1d(in_channels=64, out_channels=64,
                      kernel_size=5, stride=2, padding=2),  # torch.Size([128, 64, 32])
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出大小：torch.Size([128, 64, 16])
            nn.Flatten(),  # 输出大小：torch.Size([128, 1024]) 展平第一维及之后
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 16, out_features=32),
            nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.cnn1d(x)
        x = self.fc(x)
        return x

config = {
    "name": "1dcnn",
    "path": None,  # 数据集路径
    "input_size": 864,  # 特征长度
    "number": 1000,  # 每类样本数
    "dropout": 0.227,  # dropout率
    "batch_size": 64,  # 批量大小
    "normal": False,  # 数据集是否归一化
    "rate": [0.5, 0.25, 0.25],  # 训练集:验证集:测试集
    "enc": False,  # 是否采用数据增强
    "enc_step": 28,  # 数据增强步长
    # "is_shuffle": True, # 数据集是否随机
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 设备
    "lr": 0.001,  # 学习率
    "epochs": 30,
    "best_model_path": r"./models/1dcnn_v1.pt",
    "show": True,
    "tune": False,
}


def train_1dcnn_v1(config, train_iter, valid_iter):
    net = CNN1D(config["dropout"])
    net.to(config["device"])
    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=config["lr"])

    train.train(net, train_iter, valid_iter, loss, updater, config)

def test_1dcnn_v1(config, test_iter):
    pt = torch.load(config["best_model_path"])
    net = CNN1D(config["dropout"])
    net.load_state_dict(pt["model_state_dict"])
    net.to(config["device"])

    analyze.analyze(net, test_iter, config)

def train_or_test(config, train=None, test=None):
    train_iter, test_iter, valid_iter = load_dataset.load_1Ddata(config)

    if train:
        train_1dcnn_v1(config, train_iter, valid_iter)
    if test:
        test_1dcnn_v1(config, test_iter)



if __name__ == "__main__":
    path = r"./datasets/12k_DE_data/0HP/"
    config["path"] = path
    train_or_test(config, train=True, test=True)



