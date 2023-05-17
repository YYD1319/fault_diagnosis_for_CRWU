# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.optim as optim
from utils.train import train
from utils.train import analyze
from utils.signal_process import load_dataset

from models import _1dcnn

def train_1dcnn_v1(config, train_iter, valid_iter, load_existing_model):
    if load_existing_model:
        pt = torch.load(config["best_model_path"])
        net = _1dcnn.CNN1D(config["dropout"])
        net.load_state_dict(pt["model_state_dict"])
    else:
        net = _1dcnn.CNN1D(config["dropout"])
    net.to(config["device"])
    loss = nn.CrossEntropyLoss(reduction="none")
    updater = optim.Adam(net.parameters(), lr=config["lr"])

    train.train(net, train_iter, valid_iter, loss, updater, config)

def test_1dcnn_v1(config, test_iter):
    pt = torch.load(config["best_model_path"])
    net = _1dcnn.CNN1D(config["dropout"])
    net.load_state_dict(pt["model_state_dict"])
    net.to(config["device"])

    analyze.analyze(net, test_iter, config)

def train_or_test(config, train=None, test=None, load_existing_model=True):
    train_iter, test_iter, valid_iter = load_dataset.load_1Ddata(config)

    if train:
        train_1dcnn_v1(config, train_iter, valid_iter, load_existing_model=load_existing_model)
    if test:
        test_1dcnn_v1(config, test_iter)



if __name__ == "__main__":
    path = r"../datasets/48K_DE_data/0HP/"
    config = _1dcnn.config_1dcnn
    config["path"] = path
    train_or_test(config, train=False, test=True, load_existing_model=True)



