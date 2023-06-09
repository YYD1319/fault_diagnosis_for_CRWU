import os
import sys
import numpy as np
import pywt

# predict
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from collections import Counter

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn import preprocessing

from models import _1dcnn
from models import _2dcnn
from models import lstm

from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from utils.signal_process import load_dataset, wavelets_trans
from utils.signal_analysis import time_domain_analysis, frequency_characteristics, frequency_domain_analysis

from data import Data
from figure_canvas import MyFigureCanvas


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建UI加载器
        loader = QUiLoader()
        # 加载UI文件
        self.ui_path = r'./ui/mainWindow.ui'
        self.ui = loader.load(self.ui_path, self)
        self.ui.setFixedSize(self.ui.width(), self.ui.height())

        self.data_path = None  # 文件路径
        self.data = Data()  # 原始数据

        #记录子图数量
        self.fre_domain_plt_num = 0

        # 初始化 gv_visual_data 的显示
        self.gv_visual_data_content = MyFigureCanvas(width=self.ui.graphicsView.width() / 101,
                                                     height=self.ui.graphicsView.height() / 101,
                                                     xlim=(0, 2 * np.pi),
                                                     ylim=(-1, 1))  # 实例化一个FigureCanvas
        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene.addWidget(
            self.gv_visual_data_content)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的
        self.ui.graphicsView.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView
        self.ui.graphicsView.show()  # 调用show方法呈现图形

        self.gv_visual_data_content_2 = MyFigureCanvas(width=self.ui.graphicsView_2.width() / 101,
                                                     height=self.ui.graphicsView_2.height() / 101,
                                                     xlim=(0, 2 * np.pi),
                                                     ylim=(-1, 1), nrows=4)
        self.graphic_scene_2 = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene_2.addWidget(
            self.gv_visual_data_content_2)
        self.ui.graphicsView_2.setScene(self.graphic_scene_2)
        self.ui.graphicsView_2.show()

        self.gv_visual_data_content_3 = MyFigureCanvas(width=self.ui.graphicsView_3.width() / 101,
                                                       height=self.ui.graphicsView_3.height() / 101,
                                                       xlim=(0, 2 * np.pi),
                                                       ylim=(-1, 1))
        self.graphic_scene_3 = QGraphicsScene()  # 创建一个QGraphicsScene
        self.graphic_scene_3.addWidget(
            self.gv_visual_data_content_3)
        self.ui.graphicsView_3.setScene(self.graphic_scene_3)
        self.ui.graphicsView_3.show()

        # 1.读入数据
        self.ui.lineEdit_x_left.setValidator(QtGui.QIntValidator(0, 130000))
        self.ui.lineEdit_x_right.setValidator(QtGui.QIntValidator(0, 130000))
        self.ui.lineEdit_fs.setValidator(QtGui.QIntValidator(1, 480001))

        self.ui.pushButton_choose_file.clicked.connect(self.open_file)
        self.ui.pushButton_draw_raw_signal.clicked.connect(self.draw_raw_signal)
        self.ui.pushButton_load_data.clicked.connect(self.load_data)

        # 2.特侦频率
            # 输入
        self.ui.lineEdit_rotation_speed.setValidator(QtGui.QIntValidator(0, 10000))
        self.ui.lineEdit_pitch_diameter.setValidator(QtGui.QDoubleValidator(0., 100., 2))
        self.ui.lineEdit_rolling_element_diameter.setValidator(QtGui.QDoubleValidator(0., 100., 2))
        self.ui.lineEdit_contact_angle.setValidator(QtGui.QIntValidator(0, 90))
        self.ui.lineEdit_rolling_element_num.setValidator(QtGui.QIntValidator(0, 100))

            # 计算特征频率
        self.ui.pushButton_caculate_fre_charact.clicked.connect(self.caculate_fre_charact)
        self.ui.pushButton_clear_fre_charact.clicked.connect(self.clear_fre_charact)

        # 3.时域分析
        self.ui.pushButton_caculate_time_domain.clicked.connect(self.time_domain_analysis)

        # 4.频域分析
        self.ui.pushButton_FFT_plt.clicked.connect(self.plt_FFT)
        self.ui.pushButton_power_spectrum_plt.clicked.connect(self.plt_power_spectrum)
        self.ui.pushButton_cepstrum_plt.clicked.connect(self.plt_cepstrum)
        self.ui.pushButton_spectrum_plt.clicked.connect(self.plt_envelope_spectrum)
        self.ui.pushButton_clear_fre_domian.clicked.connect(self.clear_fre_domian)

        # 5.时频分析
        self.ui.pushButton_plt_wavelets_trans.clicked.connect(self.plt_wavelets_trans)
            # 根据选项卡显示隐藏画布2/3
        self.ui.toolBox.currentChanged.connect(self.tab_changed)

        # 6.故障诊断
        self.ui.pushButton_predict_1dcnn.clicked.connect(self.predict_1dcnn)
        self.ui.pushButton_predict_2dcnn.clicked.connect(self.predict_2dcnn)
        self.ui.pushButton_predict_LSTM.clicked.connect(self.predict_LSTM)


    def open_file(self):
        data_path, filetype = QFileDialog.getOpenFileName(self.ui,
                                                          "选取文件",
                                                          "./",
                                                          "mat Files (*.mat)")  # 设置文件扩展名过滤,注意用双分号间隔

        if data_path != '':
            self.ui.label_file_name.setText(os.path.basename(data_path))
            self.data_path = data_path

    def draw_raw_signal(self):
        # text = self.line_edit.text()
        try:
            self.gv_visual_data_content.fig.axes[0].clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
            # self.gv_visual_data_content.axes = self.fig.add_subplot(111)
            self.gv_visual_data_content.fig = time_domain_analysis.plt_raw_signal(self.gv_visual_data_content.fig,
                                                                              self.data.raw_data)
            self.gv_visual_data_content.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)

    def load_data(self):
        self.data.fs = int(self.ui.lineEdit_fs.text())
        self.data.sampling_period = 1.0 / self.data.fs
        try:
            self.data.raw_data = load_dataset.data_acquision(self.data_path)
            x_left = int(self.ui.lineEdit_x_left.text())
            x_right = int(self.ui.lineEdit_x_right.text())
            if x_left < x_right < self.data.raw_data.size:
                self.data.raw_data = self.data.raw_data[x_left: x_right + 1]
        except Exception as e:
            print(e)

    def caculate_fre_charact(self):
        try:
            self.data.rolling_element_num = int(self.ui.lineEdit_rolling_element_num.text())
            self.data.pitch_diameter = float(self.ui.lineEdit_pitch_diameter.text())
            self.data.rolling_element_diameter = float(self.ui.lineEdit_rolling_element_diameter.text())
            self.data.contact_angle = int(self.ui.lineEdit_contact_angle.text())
            self.data.rotation_speed = int(self.ui.lineEdit_rotation_speed.text())

            self.data.FTF, self.data.BPFI, self.data.BPFO, self.data.BSF \
                = frequency_characteristics.bearing_frequencies(self.data.rolling_element_num,
                                                                self.data.pitch_diameter,
                                                                self.data.rolling_element_diameter,
                                                                self.data.contact_angle,
                                                                self.data.rotation_speed
                                                                )

            self.ui.label_FTF_res.setText(f'{self.data.FTF: .2f}')
            self.ui.label_BPFI_res.setText(f'{self.data.BPFI: .2f}')
            self.ui.label_BPFO_res.setText(f'{self.data.BPFO: .2f}')
            self.ui.label_BSF_res.setText(f'{self.data.BSF: .2f}')
        except Exception as e:
            print(e)

    def clear_fre_charact(self):
        self.data.rolling_element_num = None  # 滚珠个数
        self.data.pitch_diameter = None  # 轴承滚道节径
        self.data.rolling_element_diameter = None  # 滚珠直径
        self.data.contact_angle = None  # 轴承接触角
        self.data.rotation_speed = None  # 内圈转速

        self.data.FTF = None  # 保持架频率
        self.data.BPFI = None  # 滚动体通过内圈频率
        self.data.BPFO = None  # 滚动体通过外圈频率
        self.data.BSF = None  # 滚动体自转频率

        self.ui.lineEdit_rolling_element_num.setText('')
        self.ui.lineEdit_pitch_diameter.setText('')
        self.ui.lineEdit_rolling_element_diameter.setText('')
        self.ui.lineEdit_contact_angle.setText('')
        self.ui.lineEdit_rotation_speed.setText('')

        self.ui.label_FTF_res.setText('')
        self.ui.label_BPFI_res.setText('')
        self.ui.label_BPFO_res.setText('')
        self.ui.label_BSF_res.setText('')

    def time_domain_analysis(self):
        try:
            time_domian_dict = time_domain_analysis.time_domain_analysis(self.data.raw_data)
            self.ui.label_max_value_res.setText(f'{time_domian_dict["最大值"]: .2f}')
            self.ui.label_min_value_res.setText(f'{time_domian_dict["最小值"]: .2f}')
            self.ui.label_max_value_res_2.setText(f'{time_domian_dict["峰值"]: .2f}')
            self.ui.label_peak_value_res.setText(f'{time_domian_dict["峰峰值"]: .2f}')
            self.ui.label_mean_value_res.setText(f'{time_domian_dict["均值"]: .2f}')
            self.ui.label_abs_mean_value_res.setText(f'{time_domian_dict["平均幅值"]: .2f}')
            self.ui.label_rms_value_res.setText(f'{time_domian_dict["均方根"]: .2f}')
            self.ui.label_sra_value_res.setText(f'{time_domian_dict["方根幅值"]: .2f}')
            self.ui.label_variance_value_res.setText(f'{time_domian_dict["方差"]: .2f}')
            self.ui.label_kurtosis_value_res.setText(f'{time_domian_dict["峭度"]: .2f}')
            self.ui.label_skewness_value_res.setText(f'{time_domian_dict["偏度"]: .2f}')
            self.ui.label_crest_factor_res.setText(f'{time_domian_dict["峭度指标"]: .2f}')
            self.ui.label_form_factor_res.setText(f'{time_domian_dict["波形因子"]: .2f}')
            self.ui.label_impulse_factor_res.setText(f'{time_domian_dict["峰值因子"]: .2f}')
            self.ui.label_margin_factor_res.setText(f'{time_domian_dict["脉冲因子"]: .2f}')
        except Exception as e:
            print(e)

    def plt_FFT(self):
        self.fre_domain_plt_num += 1
        try:
            # self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图

            self.gv_visual_data_content_2.fig = frequency_domain_analysis.plt_FFT(self.gv_visual_data_content_2.fig,
                                                                            self.data.raw_data,
                                                                            self.data.sampling_period,
                                                                            self.fre_domain_plt_num)
            self.gv_visual_data_content_2.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)

        self.ui.pushButton_FFT_plt.setEnabled(False)
    def plt_power_spectrum(self):
        self.fre_domain_plt_num += 1
        try:
            # self.gv_visual_data_content_2.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
            self.gv_visual_data_content_2.fig = frequency_domain_analysis.plt_power_spectrum(self.gv_visual_data_content_2.fig,
                                                                            self.data.raw_data,
                                                                            self.data.sampling_period,
                                                                                self.fre_domain_plt_num)
            self.gv_visual_data_content_2.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)
        self.ui.pushButton_power_spectrum_plt.setEnabled(False)

    def plt_cepstrum(self):
        self.fre_domain_plt_num += 1
        try:
            # self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
            self.gv_visual_data_content_2.fig = frequency_domain_analysis.plt_cepstrum(self.gv_visual_data_content_2.fig,
                                                                            self.data.raw_data,
                                                                            self.data.sampling_period,
                                                                                self.fre_domain_plt_num)
            self.gv_visual_data_content_2.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)
        self.ui.pushButton_cepstrum_plt.setEnabled(False)


    def plt_envelope_spectrum(self):
        self.fre_domain_plt_num += 1
        try:
            # self.gv_visual_data_content_2.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
            self.gv_visual_data_content_2.fig = frequency_domain_analysis.plt_envelope_spectrum(self.gv_visual_data_content_2.fig,
                                                                            self.data.raw_data,
                                                                            self.data.sampling_period,
                                                                                self.fre_domain_plt_num)
            self.gv_visual_data_content_2.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)
        self.ui.pushButton_spectrum_plt.setEnabled(False)

    def clear_fre_domian(self):
        for i in range(self.fre_domain_plt_num):
            self.gv_visual_data_content_2.fig.axes[i].clear()
            self.gv_visual_data_content_2.fig.axes[i].axis('off')
        self.gv_visual_data_content_2.draw()
        self.fre_domain_plt_num = 0

        self.ui.pushButton_FFT_plt.setEnabled(True)
        self.ui.pushButton_power_spectrum_plt.setEnabled(True)
        self.ui.pushButton_cepstrum_plt.setEnabled(True)
        self.ui.pushButton_spectrum_plt.setEnabled(True)

    def tab_changed(self, index):
        if index == 3:
            self.ui.graphicsView_2.show()
            self.ui.graphicsView_3.hide()
        if index == 4:
            self.ui.graphicsView_2.hide()
            self.ui.graphicsView_3.show()

    def plt_wavelets_trans(self):
        data = self.data.raw_data

        N = data.size
        fs = self.data.fs
        t = np.linspace(0, N / fs, N, endpoint=False)
        wave_name = self.ui.comboBox_wave_name.currentText()
        total_scal = 256
        fc = pywt.central_frequency(wave_name)
        cparam = 2 * fc * total_scal  # 常数(n)影响图像分布区域
        scales = cparam / np.arange(total_scal, 1, -1)
        sampling_period = 1.0 / fs

        try:
            # self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
            self.gv_visual_data_content_3.fig = wavelets_trans.cwt_trans(data, t, scales, wave_name, sampling_period, fig=self.gv_visual_data_content_3.fig)
            self.gv_visual_data_content_3.draw()  # 刷新画布显示图片，否则不刷新显示
        except Exception as e:
            print(e)

    def get_np_data(self):
        data = None
        length = 864
        Samples = []
        file = loadmat(self.data_path)
        file_keys = file.keys()

        for key in file_keys:
            if 'DE' in key:  # 驱动端振动数据
                data = file[key].ravel()

        for j in range(len(data) // length):
            random_start = np.random.randint(low=0, high=(len(data) - length))
            # 每次取j至j+length长度的数据
            sample = data[j * length:(j + 1) * length]
            # sample = data[random_start:random_start + length]
            Samples.append(sample)


        Samples = np.asarray(Samples)

        # scalar = preprocessing.StandardScaler().fit(Samples)
        # Samples = scalar.transform(Samples)

        return Samples

    def get_1d_data_iter(self):
        Samples = self.get_np_data()

        Samples = Samples[:, np.newaxis, :]
        Samples = torch.tensor(Samples, dtype=torch.float)

        dataset = MyDataset(Samples)
        data_iter = DataLoader(dataset, batch_size=256, shuffle=True)

        return data_iter

    def get_2d_data_iter(self):
        Samples = self.get_np_data()

        length = 864
        N = length
        fs = self.data.fs
        t = np.linspace(0, N / fs, N, endpoint=False)
        wave_name = 'cmor3-3'
        total_scal = 256
        fc = pywt.central_frequency(wave_name)
        cparam = 2 * fc * total_scal  # 常数(n)影响图像分# 布区域
        scales = cparam / np.arange(total_scal, 1, -1)
        sampling_period = 1.0 / fs

        for i in range(len(Samples)):
            X = Samples[i].squeeze()
            fig = wavelets_trans.cwt_trans(X, t, scales, wave_name, sampling_period)

            save_path = r'../datasets/cwt_picture/temp/class/' + str(i) + '.jpg'
            fig.savefig(save_path)
            plt.close(fig)
        #
        data_transfrom = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Resize((52, 52)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(r'D:\Code\fault_diagnosis_for_CRWU\datasets\cwt_picture\temp',
                                       transform=data_transfrom)
        data_iter = DataLoader(dataset, batch_size=128, num_workers=0)

        return data_iter

    def predict(self, net, data_iter, config, is_2d=False):
        Y_hat = []
        if isinstance(net, torch.nn.Module):
            net.eval()
        with torch.no_grad():
            if is_2d:
                for X, _ in data_iter:
                    X = X.to(config["device"])
                    y_hat = net(X)
                    Y_hat.append(y_hat)
            else:
                for X in data_iter:
                    X = X.to(config["device"])
                    y_hat = net(X)
                    Y_hat.append(y_hat)
        Y_hat = torch.cat(Y_hat, dim=0).cpu().numpy().argmax(1)

        return Y_hat

    def predict_1dcnn(self):
        data_iter = self.get_1d_data_iter()

        config = _1dcnn.config_1dcnn

        pt = torch.load(config["best_model_path"])
        net = _1dcnn.CNN1D(config["dropout"])
        net.load_state_dict(pt["model_state_dict"])
        net.to(config["device"])

        Y_hat = self.predict(net, data_iter, config)

        target_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'Normal']
        self.ui.label_predict_1dcnn.setText(target_names[Counter(Y_hat).most_common(1)[0][0]])
        # print(Y_hat.shape)
        # print(Y_hat)
        # print(type(Y_hat))

    def predict_2dcnn(self):
        data_iter = self.get_2d_data_iter()

        config = _2dcnn.config_2dcnn

        pt = torch.load(config["best_model_path"])
        net = _2dcnn.CNN2D()
        net.load_state_dict(pt["model_state_dict"])
        net.to(config["device"])

        Y_hat = self.predict(net, data_iter, config, is_2d=True)

        target_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'Normal']
        self.ui.label_predict_2dcnn.setText(target_names[Counter(Y_hat).most_common(1)[0][0]])

    def predict_LSTM(self):
        data_iter = self.get_1d_data_iter()

        config = lstm.config_lstm


        pt = torch.load(config["best_model_path"])
        net = lstm.LSTM(config["input_size"], config["hidden_dim"], config["layer_dim"], config["num_classes"], config["dropout"])
        net.load_state_dict(pt["model_state_dict"])
        net.to(config["device"])

        Y_hat = self.predict(net, data_iter, config)

        target_names = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'Normal']
        self.ui.label_predict_LSTM.setText(target_names[Counter(Y_hat).most_common(1)[0][0]])



if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    stats = MainWindow()
    stats.ui.show()
    app.exec_()
