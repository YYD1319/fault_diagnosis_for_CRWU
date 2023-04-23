import torch
import matplotlib.pyplot as plt
from IPython import display
from sklearn.metrics import confusion_matrix
import time
import numpy as np
from ray import tune


def draw_curve(x_val, y_val, x_label, y_label, title, x2_val=None, y2_val=None, le=None, figsize=(7, 5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_val, y_val, label='2')
    if x2_val and y2_val:
        plt.semilogy(x2_val, y2_val, linestyle=':', label='1')
    plt.legend(le)
    plt.title(title)
    plt.show()
    plt.close()


def accuracy(y_hat, y):
    return int((y_hat.argmax(1) == y).sum())


def evaluate(net, valid_iter, loss_func, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        loss_sum, acc, n = 0, 0, 0
        for X, y in valid_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss_sum += loss_func(y_hat, y).sum()
            acc += accuracy(y_hat, y)
            n += len(y)
        return loss_sum / n, acc / n


def train_epoch(net, train_iter, loss_func, updater, device):
    loss_sum, acc, n = 0, 0, 0
    if isinstance(net, torch.nn.Module):
        net.train()
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        updater.zero_grad()

        y_hat = net(X)
        # print(y_hat.shape, y.shape) [128, 10], [128, 10]

        loss = loss_func(y_hat, y)

        loss.mean().backward()
        updater.step()
        loss_sum += loss.sum()
        n += len(y)
        acc += accuracy(y_hat, y)
        # print(acc / len(y))
    return loss_sum / n, acc / n


def train(net, train_iter, valid_iter, loss, updater, cfg):
    net.to(cfg["device"])
    print("epoch\ttrain_loss\ttest_loss\ttrain_acc\ttest_acc\ttrain_time")
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []
    best_loss = float('inf')
    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater, cfg["device"])
        test_loss, test_acc = evaluate(net, valid_iter, loss, cfg["device"])
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': updater.state_dict(),
                'best_loss': best_loss
            }, cfg["best_model_path"])

        train_time = (time.time() - epoch_start)
        if cfg["show"]:
            print("%d\t\t%f\t%f\t%f\t%f\t%.6fs" % (epoch + 1, train_loss, test_loss, train_acc, test_acc, train_time))

        train_loss_list.append(train_loss.cpu().detach().numpy())
        test_loss_list.append(test_loss.cpu().detach().numpy())
        train_acc_list.append(np.array(train_acc))
        test_acc_list.append(np.array(test_acc))

        # test_log_list.append(test_log)

    # plt.subplots_adjust(wspace=0.2,hspace=0.5)
    if cfg["show"]:
        draw_curve(range(1, cfg["epochs"] + 1), train_loss_list, 'epochs', 'loss', 'training and validation loss(' + cfg["name"] + ')',
                   range(1, cfg["epochs"] + 1), test_loss_list,
                   ['train', 'valid'])
        draw_curve(range(1, cfg["epochs"] + 1), train_acc_list, 'epochs', 'acc', 'training and validation accuracy(' + cfg["name"] + ')',
                   range(1, cfg["epochs"] + 1), test_acc_list,
                   ['train', 'valid'])
    if cfg["tune"]:
        tune.report(loss=sum(test_loss_list) / len(train_loss_list) , accuracy=sum(test_acc_list) / len(train_loss_list))
