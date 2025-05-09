import numpy as np

from MultiChannel import MultiChannel, MultiChannelConverter

sRGB = MultiChannel('sRGB', 3)
BT2020 = MultiChannel('BT2020', 3)

if __name__ == '__main__':
    # 构建 sRGB 和 BT2020 到 XYZ 的转换矩阵
    sRGB.build_from_XYZ(np.array([
       [0.4900 , 0.31000, 0.20000],
       [0.17697, 0.81240, 0.01063],
       [0.00000, 0.01000, 0.99000] 
    ])) # 数据来源：维基百科
    sRGB.plot()

    BT2020.build_from_XYZ(np.array([
       [0.636958, 0.144617, 0.168881],
       [0.262700, 0.677998, 0.059302],
       [0.000000, 0.028073, 1.060985]
    ])) # 数据来源：ITU-R BT.2020 标准 计算得到
    BT2020.plot()

    # 构建转换器
    converter = MultiChannelConverter(sRGB, BT2020)
    # 计算损失
    loss = converter.loss()
    print(f"Loss: {loss}")
