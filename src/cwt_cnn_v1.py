import torch
import torch.nn as nn
import torch.optim as optim
from utils.train import train
from utils.train import analyze
from utils.signal_process import load_dataset
from utils.train.train import draw_kfold_curve

from models import _2dcnn

def train_cwt_cnn_v1(config, train_iter, valid_iter, kfold=False, train_valid_dataset=None, load_existing_model=False):
    if load_existing_model:
        pt = torch.load(config["best_model_path"])
        net = _2dcnn.CNN2D()
        net.load_state_dict(pt["model_state_dict"])
        net.to(config["device"])
    else:
        net = _2dcnn.CNN2D()
    # net.apply(init_constant)

    loss = nn.CrossEntropyLoss(reduction="none")  # 公式是什么
    updater = optim.Adam(net.parameters(), lr=config["lr"])  # 原理是什么
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []
    if kfold:
        for train_iter, valid_iter, i in load_dataset.get_kfold_train_vaalid_iter(train_valid_dataset, config["k"], config["batch_size"], shuffle=True):
            print("第{}折------------------------------------------------------------------------".format(i + 1))
            train_loss, test_loss, train_acc, test_acc = train.train(net, train_iter, valid_iter, loss, updater, config, kfold=kfold)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

        draw_kfold_curve(config["k"], range(1, config["kfold_epochs"] + 1), train_loss_list, 'epochs', 'loss',
                   'Training and validation loss(' + config["name"] + ')',
                   range(1, config["kfold_epochs"] + 1), test_loss_list,
                   ['train', 'valid'])
        draw_kfold_curve(config["k"], range(1, config["kfold_epochs"] + 1), train_acc_list, 'epochs', 'acc',
                   'Training and validation accuracy(' + config["name"] + ')',
                   range(1, config["kfold_epochs"] + 1), test_acc_list,
                   ['train', 'valid'])
    else:
        train.train(net, train_iter, valid_iter, loss, updater, config)

def test_cwt_cnn_v1(config, test_iter):
    pt = torch.load(config["best_model_path"])
    net = _2dcnn.CNN2D()
    net.load_state_dict(pt["model_state_dict"])
    net.to(config["device"])

    analyze.analyze(net, test_iter, config)
def train_or_test(cfg, kfold=False, train=False, test=False, load_existing_model=False):
    if kfold:
        train_valid_dataset, test_dataset = load_dataset.get_kfold_dataset(cfg["dataset_path"], cfg["kfold_rate"], cfg["size"])
        if train:
            train_cwt_cnn_v1(cfg, train_iter=None, valid_iter=None, kfold=kfold, train_valid_dataset=train_valid_dataset, load_existing_model=load_existing_model)

            # test_iter = load_dataset.get_kfold_test_iter(test_dataset, cfg["batch_size"])
            # test_cwt_cnn_v1(cfg, test_iter)

    else:
        # 读取数据集
        train_iter, valid_iter, test_iter = load_dataset.load_2Ddata(cfg["dataset_path"], cfg["rate"],
                                                                     cfg["batch_size"], cfg["size"])

        if train:

            train_cwt_cnn_v1(cfg, train_iter, valid_iter, load_existing_model=load_existing_model)
        if test:
            cfg["rate"] = [0, 0, 1]
            train_iter, valid_iter, test_iter = load_dataset.load_2Ddata(cfg["dataset_path"], cfg["rate"],
                                                                         cfg["batch_size"], cfg["size"])
            test_cwt_cnn_v1(cfg, test_iter)

if __name__ == "__main__":
    # train_path = r"../datasets/cwt_picture/0HP/cmor3-3/train"
    # valid_path = r"../datasets/cwt_picture/0HP/cmor3-3/valid"
    # test_path = r"../datasets/cwt_picture/0HP/cmor3-3/test"

    config = _2dcnn.config_2dcnn

    config["dataset_path"] = r"../datasets/cwt_picture/cmor3-3_4000_train/"
    config["best_model_path"] = r"../models/gary_largeKernel_{}_kfold_{}_best_model_12k.pt".format(config["name"], config["k"])


    # config["train_path"] = train_path
    # config["valid_path"] = valid_path
    # config["test_path"] = test_path

    train_or_test(config, kfold=True, train=True, test=True, load_existing_model=False)


