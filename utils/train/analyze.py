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


def draw_confusion(Y_hat, Y, target_names, name):
    con_mat = confusion_matrix(Y_hat, Y)
    # print(con_mat)
    plt.imshow(con_mat, cmap=plt.cm.Reds)
    indices = range(len(con_mat))
    plt.xticks(indices, target_names, rotation=45)
    plt.yticks(indices, target_names)
    plt.colorbar()
    plt.xlabel('pre')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.title("confusion matrix(" + name + ")")
    plt.show()
    plt.close()

def draw_classfi_report(Y_hat, Y, target_names, name):

    report_dict = classification_report(Y, Y_hat, target_names=target_names, output_dict=True)
    # 获取每个类别的 Precision、Recall 和 F1 Score
    precision = [report_dict[name]['precision'] for name in target_names]
    recall = [report_dict[name]['recall'] for name in target_names]
    f1_score = [report_dict[name]['f1-score'] for name in target_names]

    # 绘制柱状图
    fig, ax = plt.subplots()
    x = range(len(target_names))
    bar_width = 0.2
    opacity = 0.8

    ax.bar(x, precision, bar_width, alpha=opacity, color='r', label='Precision')
    ax.bar([i + bar_width for i in x], recall, bar_width, alpha=opacity, color='g', label='Recall')
    ax.bar([i + 2 * bar_width for i in x], f1_score, bar_width, alpha=opacity, color='b', label='F1 Score')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report(' + name + ')')
    ax.set_xticks([i + bar_width for i in x])
    ax.set_xticklabels(target_names, rotation=45)
    ax.legend(bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def analyze(net, test_iter, cfg):
    Y_hat, Y = get_predict_labels(net, test_iter, cfg["device"])

    target_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'Normal']

    draw_confusion(Y_hat, Y, target_names, cfg["name"])

    draw_classfi_report(Y_hat, Y, target_names, cfg["name"])

    print(classification_report(Y, Y_hat))
