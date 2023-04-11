# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.optim as optim
import train
import load_dataset


class CNN1D(nn.Module):
    def __init__(self, dropout):
        super(CNN1D, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32,
                      kernel_size=20, stride=8, padding=598),  # torch.Size([128, 32, 256]) [N, C_out, L_out]
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # torch.Size([128, 32, 64])
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=2, padding=2),  # torch.Size([128, 64, 32])
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出大小：torch.Size([128, 64, 16])
            nn.Flatten(),  # 输出大小：torch.Size([128, 1024]) 展平第一维及之后
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024, out_features=32),
            nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.cnn1d(x)
        x = self.fc(x)
        return x


class Config:
    def __init__(self):
        # self.path = r"./datasets/48k_DE_data/0HP"  # 数据集路径
        self.path = None
        self.input_size = 864  # 特征长度
        self.number = 1000  # 每类样本数
        self.dropout = 0.5  # dropout率
        self.batch_size = 128  # 批量大小
        self.normal = True  # 数据集是否归一化
        self.rate = [0.5, 0.25, 0.25]  # 训练集:验证集:测试集
        self.enc = True  # 是否采用数据增强
        self.enc_step = 28  # 数据增强步长
        # self.is_shuffle = True  # 数据集是否随机
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
        self.lr = 0.001  # 学习率
        self.epochs = 10
        self.best_model_path = r"./models/_1dcnn_v1.pt"


def _1dcnn_v1(path):
    cfg = Config()
    cfg.path = path

    net = CNN1D(cfg.dropout)
    net.to(cfg.device)
    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=cfg.lr)

    train_iter, test_iter, valid_iter = load_dataset.load_1Ddata(cfg)

    train.train(net, train_iter, valid_iter, loss, updater, cfg)


if __name__ == "__main__":
    path = r"./datasets/48k_DE_data/0HP"
    _1dcnn_v1(path)
