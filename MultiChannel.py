import numpy as np

from plot import plot_color_point

class MultiChannel:
    """
    多通道颜色空间类，此处由于题目设置，只考虑不小于 3 通道的情况。
    """
    def __init__(self, color_space: str, num_channels: int):
        assert num_channels >= 3, "通道数必须大于等于 3"

        self.name = color_space
        self.num_channels = num_channels
        self.M = np.zeros((3, num_channels))

    def __str__(self):
        return self.name

    def to_XYZ(self, color: np.ndarray):
        """
        将该颜色空间中的颜色无损转换至 XYZ 颜色空间。

        Parameters:
            color (np.ndarray): 输入颜色，形状为 [batch, num_channels]。

        Returns:
            np.ndarray: 输出颜色，形状为 [batch, 3]。
        """

        assert color.shape[1] == self.num_channels, f"{self.name} 颜色空间的通道数必须为 {self.num_channels}"
        return color @ self.M

    def from_XYZ(self, XYZ: np.ndarray):
        """
        将 XYZ 颜色空间中的颜色转换至该颜色空间。

        Parameters:
            XYZ (np.ndarray): 输入颜色，形状为 [batch, 3]。

        Returns:
            np.ndarray: 输出颜色，形状为 [batch, num_channels]。
        """
        assert XYZ.shape[1] == 3, "XYZ 颜色空间的通道数必须为 3"
        if self.num_channels == 3:
            # 三通道，直接乘以逆矩阵（这是无损转换）
            return XYZ @ np.linalg.inv(self.M)
        else:
            # TODO: 非三通道，需要更复杂的转换逻辑 (For Task 2)
            return np.ones((XYZ.shape[0], self.num_channels))

    def build_from_xyY(self, xyY: np.ndarray):
        """
        从 xyY 颜色空间输入构建矩阵M。

        Parameters:
            xyY (np.ndarray): 该颜色空间的标准色的 xyY 坐标，形状为 [num_channels, 3]。
        """
        assert xyY.shape[0] == self.num_channels, f"{self.name} 颜色空间的通道数必须为 {self.num_channels}"
        assert xyY.shape[1] == 3, "xyY 坐标需要 3 个参数"

        # 转换为 XYZ 颜色空间
        XYZ = np.zeros((self.num_channels, 3))
        x = xyY[:, 0]
        y = xyY[:, 1]
        Y = xyY[:, 2]
        XYZ[:, 0] = x * Y / y
        XYZ[:, 1] = Y
        XYZ[:, 2] = (1 - x - y) * Y / y
        # 构建矩阵M
        self.M = XYZ

    def build_from_xy(self, xy: np.ndarray, white_point: np.ndarray=np.array([0.3127, 0.3290])):
        """
        从 xy 颜色空间输入构建矩阵M。

        Parameters:
            xy (np.ndarray): 该颜色空间的标准色的 xy 坐标，形状为 [num_channels, 2]。
            white_point (np.ndarray): 该颜色空间的白点坐标，形状为 [2]。

        求解过程参见https://fujiwaratko.sakura.ne.jp/infosci/colorspace/rgb_xyz_e.html
        """
        assert xy.shape[0] == self.num_channels, f"{self.name} 颜色空间的通道数必须为 {self.num_channels}"
        assert xy.shape[1] == 2, "xy 坐标需要 2 个参数"
        assert white_point.shape == (2,), "白点坐标需要 2 个参数"

        # 计算归一化的矩阵M
        M = np.zeros((3, self.num_channels))
        M[:, 0] = xy[:, 0]
        M[:, 1] = xy[:, 1]
        M[:, 2] = 1 - xy[:, 0] - xy[:, 1]

        # 计算 Y=1 的白色点坐标
        white_point_scale = 1 / white_point[1]
        white_point_XYZ = np.array([
            [white_point[0] * white_point_scale],
            [1],
            [(1 - white_point[0] - white_point[1]) * white_point_scale]
        ]) # 形状为 [3, 1]

        # 根据白色点坐标计算各点颜色实际坐标（非归一化）
        scale = np.linalg.inv(M.T) @ white_point_XYZ # 形状为 [3, 1]
        M[0, :] *= scale[0, 0]
        M[1, :] *= scale[1, 0]
        M[2, :] *= scale[2, 0]

        # 构建矩阵M
        self.M = M

    def build_from_XYZ(self, XYZ: np.ndarray):
        """
        从 XYZ 颜色空间输入构建矩阵M。

        Parameters:
            XYZ (np.ndarray): 该颜色空间的标准色的 XYZ 坐标，形状为 [num_channels, 3]。
        """
        assert XYZ.shape[0] == self.num_channels, f"{self.name} 颜色空间的通道数必须为 {self.num_channels}"
        assert XYZ.shape[1] == 3, "XYZ 坐标需要 3 个参数"

        # 构建矩阵M
        self.M = XYZ.T

    def plot(self, resolution: int=100, Y: float=None):
        """
        在 xyY 空间中绘制该颜色空间的颜色图，用于检验参数正确性。

        Parameters:
            resolution (int): 分辨率，表示平均每个通道的采样数。
            Y (float): 亮度，默认为 1.0。
        """
        # 生成所有采样点
        points = [np.linspace(0, 1, resolution) for _ in range(self.num_channels)]
        points = np.meshgrid(*points)
        points = np.stack(points, axis=-1).reshape(-1, self.num_channels)
        # 转换为 XYZ 颜色空间，过滤与 Y 相差较大的点
        XYZ = self.to_XYZ(points)
        if Y is not None:
            mask = np.abs(XYZ[:, 1] - Y) < 0.1
            XYZ = XYZ[mask]
        # 转换为 xyY 颜色空间的坐标
        sum_XYZ = np.sum(XYZ, axis=1).clip(min=1e-6)
        x = XYZ[:, 0] / sum_XYZ
        y = XYZ[:, 1] / sum_XYZ
        # 绘制颜色图
        plot_color_point(XYZ, title=self.name, xs=x, ys=y)


class MultiChannelConverter:
    """
    多通道颜色空间转换器，用于将一个颜色空间中的颜色转换至另一个颜色空间。
    """
    def __init__(self, src: MultiChannel, dst: MultiChannel):
        self.src = src
        self.dst = dst

    def solve_overflow(self, color: np.ndarray):
        """
        解决颜色溢出问题，将超出范围的颜色值映射回范围内。

        Parameters:
            color (np.ndarray): 输入颜色，形状为 [batch, num_channels]，可能包含[0, 1]以外的数值。

        Returns:
            np.ndarray: 输出颜色，形状为 [batch, num_channels]，所有值都在[0, 1]范围内。
        """
        # TODO: 此处的处理方法是一个很粗糙的处理，需要进一步优化
        return np.clip(color, 0.0, 1.0)

    def convert(self, color_XYZ: np.ndarray):
        """
        将颜色从源颜色空间转换至目标颜色空间。
        此处只是一个简单的示例，之后可能需要更复杂的转换逻辑。

        Parameters:
            color_XYZ (np.ndarray): 输入颜色的 XYZ 坐标，形状为 [batch, src_channels]。

        Returns:
            np.ndarray: 输出颜色的 XYZ 坐标，形状为 [batch, dst_channels]。
        """
        assert color_XYZ.shape[1] == 3, f"XYZ 通道数必须为 3"

        # 转换为目标颜色空间（无损转换）
        dst_color = self.dst.from_XYZ(color_XYZ)
        # 解决颜色溢出问题（有损）
        dst_color = self.solve_overflow(dst_color)
        # 转换回 XYZ 颜色空间
        dst_color_XYZ = self.dst.to_XYZ(dst_color)

        return dst_color_XYZ

    def loss(self, resolution: int=100, strategy: str='uniform'):
        """
        计算转换损失。

        Parameters:
            resolution (int): 分辨率，表示平均每个通道的采样数。
            strategy (str): 测试数据生成策略，'uniform' 表示均匀采样，'random' 表示随机采样。

        Returns:
            float: 转换损失。
        """
        if strategy == 'uniform':
            # 均匀采样
            points = [np.linspace(0, 1, resolution) for _ in range(self.src.num_channels)]
            points = np.meshgrid(*points)
            points = np.stack(points, axis=-1).reshape(-1, self.src.num_channels)
        elif strategy == 'random':
            # 随机采样
            points = np.random.rand(resolution**self.src.num_channels, self.src.num_channels)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # 将输入颜色转换至 XYZ 颜色空间
        src_color_XYZ = self.src.to_XYZ(points)
        # 转换至目标颜色空间（表示为 XYZ 坐标）
        dst_color_XYZ = self.convert(src_color_XYZ)

        # 计算损失（转换前后的欧氏距离平方的均值作为损失）
        loss = np.mean(np.sum((src_color_XYZ - dst_color_XYZ)**2, axis=1))

        return loss
        
