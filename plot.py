import numpy as np
import matplotlib.pyplot as plt

from convert import to_XYZ, XYZ2RGB, BT2020_to_RGB

def plot_color_point(
    colors: np.ndarray,
    xs: np.ndarray=None,
    ys: np.ndarray=None,
    title: str='',
    axis_labels: tuple[str,str]=('x', 'y'),
    fig_size: tuple[float,float]=(10, 10),
    save_path=None,
):
    """
    绘制任意颜色点集，呈现在对应空间二维坐标空间中。

    Parameters:
        points (np.ndarray): XYZ 颜色点集，形状为 [batch, 3]。
        xs (np.ndarray): x 坐标，形状为 [batch]。
        ys (np.ndarray): y 坐标，形状为 [batch]。
        title (str): 图表标题。
        axis_labels: 图表坐标轴标签。
        fig_size (tuple): 图表大小。
        save_path (str): 保存路径，不填时不保存。
    """
    plt.figure(figsize=fig_size, dpi=100)

    if xs is None:
        xs = colors[:, 0]
    if ys is None:
        ys = colors[:, 1]

    # 转换为 RGB 用于显示颜色
    rgb_colors = XYZ2RGB(colors)

    plt.scatter(xs, ys,
                c=rgb_colors,
                s=20, 
                edgecolors='none')
    
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title if title else 'Color Space')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # 示例图：CIE 1931 xyY 颜色空间中0.1<x<0.4, 0.2<y<0.6的颜色区域

    resolution = 50
    width = 3 * resolution
    height =  4 * resolution

    x = np.linspace(0.1, 0.4, width)
    y = np.linspace(0.6, 0.2, height)
    Y = 1  # 固定 Y 坐标为 1

    # 生成 XYZ 坐标
    x_grid, y_grid = np.meshgrid(x, y)
    xyY = np.stack([x_grid, y_grid, np.full_like(x_grid, Y)], axis=-1).reshape(-1, 3)

    # 绘制颜色图
    plot_color_point(xyY, 'xyY')