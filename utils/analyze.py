from sklearn.metrics import classification_report
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_predict_labels(net, test_iter, device):
    Y_hat, Y = [], []
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            Y_hat.append(y_hat)
            Y.append(y)
    Y_hat = torch.cat(Y_hat, dim=0).cpu().numpy().argmax(1)
    Y = torch.cat(Y, dim=0).cpu().numpy()
    # print(Y.shape, Y_hat.shape)
    # print(type(Y), type(Y_hat))
    return Y_hat, Y


def draw_confusion(Y_hat, Y):
    con_mat = confusion_matrix(Y_hat, Y)
    # print(con_mat)
    classes = list(set(Y))
    classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Reds)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('pre')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.title("confusion matrix")
    plt.show()
    plt.close()


def analyze(net, test_iter, cfg):
    Y_hat, Y = get_predict_labels(net, test_iter, cfg["device"])

    draw_confusion(Y_hat, Y)

    print(classification_report(Y, Y_hat))
