import os
import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore, QtGui
from utils.signal_process import load_dataset
from utils.signal_analysis import time_domain_analysis, frequency_characteristics


# 创建一个数据类，保存原始数据和处理后的数据
class Data():
    def __init__(self):
        self.raw_data = None  # 原始数据

        # 特征频率计算变量
        self.rolling_element_num = None  # 滚珠个数
        self.pitch_diameter = None  # 轴承滚道节径
        self.rolling_element_diameter = None  # 滚珠直径
        self.contact_angle = None  # 轴承接触角
        self.rotation_speed = None  # 内圈转速

        # 特征频率计算结果
        self.FTF = None  # 保持架频率
        self.BPFI = None  # 滚动体通过内圈频率
        self.BPFO = None  # 滚动体通过外圈频率
        self.BSF = None  # 滚动体自转频率


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建UI加载器
        loader = QUiLoader()
        # 加载UI文件
        ui_file = r'./ui/mainWindow.ui'
        self.ui = loader.load(ui_file, self)

        self.file_path = None  # 文件路径
        self.data = Data()  # 原始数据

        # 设置读入数据类型
        # 原始数据
        self.ui.lineEdit_x_left.setValidator(QtGui.QIntValidator())
        self.ui.lineEdit_x_right.setValidator(QtGui.QIntValidator())
        # 特侦频率计算
        self.ui.lineEdit_rotation_speed.setValidator(QtGui.QIntValidator(0, 9999))
        self.ui.lineEdit_pitch_diameter.setValidator(QtGui.QDoubleValidator(0., 99., 2))
        self.ui.lineEdit_rolling_element_diameter.setValidator(QtGui.QDoubleValidator(0., 99., 2))
        self.ui.lineEdit_contact_angle.setValidator(QtGui.QIntValidator(0, 90))
        self.ui.lineEdit_rolling_element_num.setValidator(QtGui.QIntValidator(0, 99))

        # 读入原始数据并绘图
        self.ui.pushButton_choose_file.clicked.connect(self.open_file)
        self.ui.pushButton_draw_raw_signal.clicked.connect(self.draw_raw_signal)
        self.ui.pushButton_load_data.clicked.connect(self.load_data)

        # 计算特征频率
        self.ui.pushButton_caculate_fre_charact.clicked.connect(self.caculate_fre_charact)
        self.ui.pushButton_clear_fre_charact.clicked.connect(self.clear_fre_charact)

    def open_file(self):
        file_path, filetype = QFileDialog.getOpenFileName(self.ui,
                                                          "选取文件",
                                                          "./",
                                                          "mat Files (*.mat)")  # 设置文件扩展名过滤,注意用双分号间隔

        if file_path != '':
            self.ui.label_file_name.setText(os.path.basename(file_path))
            self.file_path = file_path

    def draw_raw_signal(self):
        # text = self.line_edit.text()
        try:
            time_domain_analysis.plt_raw_signal(self.data.raw_data)
        except Exception as e:
            print(e)

    def load_data(self):
        try:
            self.data.raw_data = load_dataset.data_acquision(self.file_path)
            x_left = int(self.ui.lineEdit_x_left.text())
            x_right = int(self.ui.lineEdit_x_right.text())
            if 0 <= x_left < x_right < self.data.raw_data.size:
                self.data.raw_data = self.data.raw_data[x_left: x_right + 1]
        except Exception as e:
            print(e)

    def caculate_fre_charact(self):
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

        self.ui.lineEdit_FTF_res.setText(f'{self.data.FTF: .2f}')
        self.ui.lineEdit_BPFI_res.setText(f'{self.data.BPFI: .2f}')
        self.ui.lineEdit_BPFO_res.setText(f'{self.data.BPFO: .2f}')
        self.ui.lineEdit_BSF_res.setText(f'{self.data.BSF: .2f}')

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

        self.ui.lineEdit_FTF_res.setText('')
        self.ui.lineEdit_BPFI_res.setText('')
        self.ui.lineEdit_BPFO_res.setText('')
        self.ui.lineEdit_BSF_res.setText('')


if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    stats = MainWindow()
    stats.ui.show()
    app.exec_()
