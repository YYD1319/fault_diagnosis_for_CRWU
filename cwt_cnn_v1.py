import torch
import torch.nn as nn
import torch.optim as optim
from utils import train
from utils.train import analyze
from utils.signal_process import load_dataset

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
    "name": "2dcnn",
    "train_path": None,  # 数据集路径
    "valid_path": None,
    "test_path": None,
    "size": 52,  # 图像大小
    "val_percentage": 0.25,  # 验证集占比
    "batch_size": 128,  # 批量大小
    "num_classes": 10,  # 分类数
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  # 设备
    "lr": 0.001,  # 学习率
    "epochs": 100,  # 训练轮数
    "best_model_path": r"./models/cwt_cnn_v1.pt",  # 最优模型保存路径
    "show": True,
    "tune": False,
}

def train_cwt_cnn_v1(config, train_iter, valid_iter):
    net = Conv_2D_v1
    # net.apply(init_constant)

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=config["lr"])

    train.train(net, train_iter, valid_iter, loss, updater, config)

def test_cwt_cnn_v1(config, test_iter):
    pt = torch.load(config["best_model_path"])
    net = Conv_2D_v1
    net.load_state_dict(pt["model_state_dict"])
    net.to(config["device"])

    analyze.analyze(net, test_iter, config)
def train_or_test(config, train=None, test=None):
    train_iter, valid_iter, test_iter = load_dataset.load_2Ddata(config)

    if train:
        train_cwt_cnn_v1(config, train_iter, valid_iter)
    if test:
        test_cwt_cnn_v1(config, test_iter)

if __name__ == "__main__":
    train_path = r"D:\Code\fault_diagnosis_for_CRWU\cwt_picture\3HP\cmor3-3\train"
    valid_path = r"D:\Code\fault_diagnosis_for_CRWU\cwt_picture\3HP\cmor3-3\valid"
    test_path = r"D:\Code\fault_diagnosis_for_CRWU\cwt_picture\3HP\cmor3-3\test"

    config["train_path"] = train_path
    config["valid_path"] = valid_path
    config["test_path"] = test_path

    train_or_test(config, train=False, test=True)


