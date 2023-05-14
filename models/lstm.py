import torch
from torch import nn
class LSTM(nn.Module):
    # https://blog.csdn.net/Cyril_KI/article/details/124283845
    def __init__(self, input_size, hidden_dim, layer_dim, output_dim, dropout):
        """
    input_size:特征长度
    hidden_dim: LSTM神经元个数
    layer_dim: LSTM的层数
    output_dim:隐藏层输出的维度(分类的数量)
    """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # LSTM神经元个数
        self.layer_dim = layer_dim  # LSTM的层数

        # LSTM ＋ 全连接层
        self.lstm = nn.LSTM(input_size, hidden_dim, layer_dim,
                            batch_first=True, dropout=0)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, inputs):
        Y, state = self.lstm(inputs, None)  # None 表示初始的 hidden state 为0
        # 选取最后一个时间点的out输出
        # print(Y.shape) [128, 1, 128]
        out = self.fc1(Y[:, -1, :])  # (batch_size, seq_len, num_directions * hidden_size)
        return out

config_lstm = {
    # self.seq_length = 1  # 序列长度
    # self.path = r"./datasets/48k_DE_data/0HP"  # 数据集路径
    "name": "lstm",
    "path": None,  # 数据集路径
    "input_size": 864,  # 特征长度
    "hidden_dim": 128,  # 隐藏单元数
    "layer_dim": 1,  # 隐藏层数
    "dropout": 0.5,  # dropout率
    "num_classes": 10,  # 分类数
    "number": 1000,  # 每类样本数
    "batch_size": 128,  # 批量大小
    "normal": False,  # 数据集是否归一化
    "rate": [0.5, 0.25, 0.25],  # 训练集:验证集:测试集
    "enc": True,  # 是否采用数据增强
    "enc_step": 28,  # 数据增强步长
    # "is_shuffle": True, # 数据集是否随机
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 设备
    "lr": 0.001,  # 学习率
    "epochs": 30,  # 训练轮数
    "best_model_path": r"../models/lstm_v1.pt",  # 最优模型保存路径
    "show": True,
    "tune": False
}