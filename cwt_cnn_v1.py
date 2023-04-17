import torch
import torch.nn as nn
import torch.optim as optim
from utils import train, load_dataset, analyze

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

config = {
    "path": None,  # 数据集路径
    "size": 52,  # 图像大小
    "val_percentage": 0.25,  # 验证集占比
    "batch_size": 128,  # 批量大小
    "num_classes": 10,  # 分类数
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 设备
    "lr": 0.001,  # 学习率
    "epochs": 100,  # 训练轮数
    "best_model_path": r"./models/cwt_cnn_smallSample.pt",  # 最优模型保存路径
    "show": True,
    "tune": False,
}

def cwt_cnn_v1(config):
    net = Conv_2D_v1
    # net.apply(init_constant)

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=config["lr"])

    train_iter, test_iter = load_dataset.load_2Ddata(config)

    train.train(net, train_iter, test_iter, loss, updater, config)



if __name__ == "__main__":
    # path = r"./cwt_picture/cmor3-3/train"
    path = r"D:\Code\fault_diagnosis_for_CRWU\cwt_picture\29_0HP_cmor3-3"
    config["path"] = path
    cwt_cnn_v1(config)

    # train_iter, test_iter = load_dataset.load_2Ddata(config)
    #
    # pt = torch.load(config["best_model_path"])
    # net = Conv_2D_v1
    # net.load_state_dict(pt["model_state_dict"])
    # net.to(config["device"])
    #
    # analyze.analyze(net, test_iter, config)
