import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
class MyFigureCanvas(FigureCanvas):
    '''
    通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
    '''

    def __init__(self, parent=None, width=10, height=5, xlim=(0, 2500), ylim=(-2, 2), dpi=100, nrows=1):
        # 创建一个Figure
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi, tight_layout=True)  # tight_layout: 用于去除画图时两边的空白

        FigureCanvas.__init__(self, self.fig)  # 初始化父类

        self.setParent(parent)
        self.fig.subplots(nrows, 1)
        # self.axes = self.fig.add_subplot(111)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        # self.axes.spines['top'].set_visible(False)  # 去掉上面的横线
        # self.axes.spines['right'].set_visible(False)
        # self.axes.set_xlim(xlim)
        # self.axes.set_ylim(ylim)
        for i in range(nrows):
            self.fig.axes[i].axis('off')