from PySide2.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QFileDialog, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtCore

import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")  # 声明使用QT5


class MyFigureCanvas(FigureCanvas):
    '''
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    '''

    def __init__(self, parent=None, width=10, height=5, xlim=(0, 2500), ylim=(-2, 2), dpi=100):
        # 创建一个Figure
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, self.fig)  # 初始化父类
        self.setParent(parent)

        self.fig.add_subplot(1, 1, 1)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        self.fig.axes[0].spines['top'].set_visible(False)  # 去掉上面的横线
        self.fig.axes[0].spines['right'].set_visible(False)
        self.fig.axes[0].set_xlim(xlim)
        self.fig.axes[0].set_ylim(ylim)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 创建UI加载器
        loader = QUiLoader()
        # 加载UI文件
        ui_file = r'./ui/test.ui'
        self.ui = loader.load(ui_file, self)

        # 初始化 gv_visual_data 的显示
        self.gv_visual_data_content = MyFigureCanvas(width=self.ui.graphicsView.width() / 101,
                                                     height=self.ui.graphicsView.height() / 101,
                                                     xlim=(0, 2 * np.pi),
                                                     ylim=(-1, 1))  # 实例化一个FigureCanvas
        self.plot_cos()

        self.ui.pushButton.clicked.connect(self.plot_sin)

    def plot_cos(self):
        x = np.arange(0, 2 * np.pi, np.pi / 100)
        y = np.cos(x)
        self.gv_visual_data_content.fig.axes[0].plot(x, y)
        self.gv_visual_data_content.fig.axes[0].set_title('cos()')

        # 加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        self.graphic_scene = QGraphicsScene()  # 创建一个QGraphicsScene

        self.graphic_scene.addWidget(
            self.gv_visual_data_content)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到放到QGraphicsScene中的

        self.ui.graphicsView.setScene(self.graphic_scene)  # 把QGraphicsScene放入QGraphicsView

        self.ui.graphicsView.show()  # 调用show方法呈现图形

    def plot_sin(self):
        x = np.arange(0, 2 * np.pi, np.pi / 100)
        y = np.sin(x)


        # self.gv_visual_data_content.axes.clear()  # 由于图片需要反复绘制，所以每次绘制前清空，然后绘图
        self.gv_visual_data_content.fig.add_subplot(2, 1, 2)

        self.gv_visual_data_content.fig.axes[1].plot(x, y)
        self.gv_visual_data_content.fig.axes[1].set_title('sin()')

        # n = len(self.gv_visual_data_content.fig.axes)
        # for i in range(n):
        #     self.gv_visual_data_content.fig.axes[i].change_geometry(n, 1, i)


        self.gv_visual_data_content.draw()  # 刷新画布显示图片，否则不刷新显示


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.ui.show()
    app.exec_()