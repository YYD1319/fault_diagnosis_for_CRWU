import torch
from torch import nn

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.cnn2d = nn.Sequential(

            # 第一层
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=0),  # 52 -3 + 1 = 50 // 52 - 5 + 1  = 48// 128 - 5 + 1
            nn.BatchNorm2d(16),
            nn.ReLU(),  # 16 * 50 * 50 // 16 * 48 *48 // 16 * 124 * 124
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 * 25 * 25 // 16 * 24 *24 // 16 * 62 * 62

            # 第二层
            nn.Conv2d(in_channels=16, out_channels=32,  # 25 - 3 + 1 = 23// 24 - 5 + 1  = 20 // 62 - 5 + 1  = 58
                      kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # 32 * 23 * 23  // 32 * 20 * 20 // 32 * 60 * 60
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 * 11 * 11 = 3872 // 32 * 10 * 10 = 3200 // 32 * 29 * 29 = 26912

            # # 第三层
            # nn.Conv2d(in_channels=16, out_channels=32,
            #           kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, out_features=128),  # linear干嘛的
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 为什么在这里dropout
            nn.Linear(128, out_features=32),  # linear干嘛的
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.cnn2d(x)
        x = self.fc(x)
        return x



config_2dcnn = {
    "name": "2dcnn",
    "dataset_path": None,  # 数据集路径
    "size": 52,  # 图像大小
    "rate": [0.5, 0.25, .25],  # 验证集占比
    "kfold_rate": 1,  # 验证集占比
    "k": 5,
    "batch_size": 256,  # 批量大小
    "num_classes": 10,  # 分类数
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 设备
    "lr": 0.001,  # 学习率
    "epochs": 5,  # 训练轮数
    "kfold_epochs": 5,
    # "best_model_path": r"../models/cwt_cnn_v1.pt",  # 最优模型保存路径
"best_model_path": r"../models/largeKernel_2dcnn_kfold_5_best_model.pt",  # 最优模型保存路径
    "show": True,
    "tune": False,
}