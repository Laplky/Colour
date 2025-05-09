import numpy as np

from MultiChannel import MultiChannel, MultiChannelConverter

RGBV = MultiChannel('RGBV', 4)
RGBCX = MultiChannel('RGBCX', 5)

if __name__ == '__main__':
    # 构建 RGBV 和 RGBCX 到 XYZ 的转换矩阵，数据来源：题目文件
    # 默认 Y 为 1，后续可能需要考虑不同的 Y 值以实现白点匹配
    RGBV.build_from_xyY(np.array([
        [0.708, 0.292, 1],
        [0.17, 0.797, 1],
        [0.14, 0.046, 1],
        [0.03, 0.6, 1]
    ]))
    RGBV.plot(resolution=40)

    RGBCX.build_from_xyY(np.array([
        [0.6948, 0.3046, 1],
        [0.2368, 0.7281, 1],
        [0.1316, 0.0712, 1],
        [0.04, 0.4, 1],
        [0.1478, 0.7326, 1]
    ]))
    RGBCX.plot(resolution=20)

    # 构建转换器
    converter = MultiChannelConverter(RGBV, RGBCX)
    # 计算损失
    loss = converter.loss()
    print(f"Loss: {loss}")