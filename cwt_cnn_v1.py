import torch
import torch.nn as nn
import torch.optim as optim
import time
import load_dataset
import train


Conv_2D_v1 = nn.Sequential(
      # 第一层
      nn.Conv2d(in_channels=3, out_channels=16,
                kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # 第二层
      nn.Conv2d(in_channels=16, out_channels=32,
                kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      # # 第三层
      # nn.Conv2d(in_channels=16, out_channels=32,
      #           kernel_size=3, stride=1, padding=0),
      # nn.BatchNorm2d(32),
      # nn.ReLU(),

      nn.Flatten(),
      nn.Linear(3872, out_features=32),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(in_features=32, out_features=10),
      nn.Softmax(dim=1)
    )


class Config:
    def __init__(self):
        self.path = None  #"/content/drive/MyDrive/graduation_design_code/cwt_picture/train"
        self.size = 52
        self.val_percentage = 0.25
        self.batch_size = 256
        self.num_classes = 10  # 分类数
        self.batch_size = 128  # 批量大小
        # self.is_shuffle = True  # 数据集是否随机
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
        self.lr = 0.001  # 学习率
        self.epochs = 100
        self.best_model_path = r"./models/cwt_cnn_v1.pt"


def cwt_cnn_v1(path):
    path_train = path
    net = Conv_2D_v1
    # net.apply(init_constant)
    config = Config()
    config.path = path_train

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=config.lr)

    train_iter, test_iter = load_dataset.load_2Ddata(config)
    train.train(net, train_iter, test_iter, loss, updater, config)

if __name__ == "__main__":
    path = r"./cwt_picture/cmor3-3/train"
    cwt_cnn_v1(path)