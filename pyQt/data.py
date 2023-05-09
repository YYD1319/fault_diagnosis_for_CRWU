class Data():
    def __init__(self):
        self.raw_data = None  # 原始数据
        self.fs = None  # 采样频率
        self.sampling_period = None

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

