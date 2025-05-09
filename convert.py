import colour
import numpy as np

def XYZ2RGB(XYZ):
    """
    使用 Colour 库将 XYZ 颜色空间转换为 sRGB 颜色空间。
    
    Parameters:
        XYZ (np.ndarray): 输入数组，形状为 [batch, 3]，表示 XYZ 颜色值。
        
    Returns:
        np.ndarray: 输出数组，形状为 [batch, 3]，表示 sRGB 颜色值（范围 [0, 1]）。
    """
    assert XYZ.shape[1] == 3, "XYZ 通道数应为 3"

    # TODO: 作为参赛代码，有可能需要自己写而不是调库

    # 定义 sRGB 色彩空间（D65 白点）
    sRGB = colour.models.RGB_COLOURSPACE_sRGB
    
    linear_rgb = colour.XYZ_to_RGB(
        XYZ, 
        colourspace=sRGB,
        illuminant_XYZ=sRGB.whitepoint,
        illuminant_RGB=sRGB.whitepoint,
        chromatic_adaptation_transform="Bradford"
    )
    
    # Gamma 校正
    gamma_rgb = colour.cctf_encoding(linear_rgb, function="sRGB")

    # 处理超出范围的RGB值
    # TODO: 这里的处理方法是一个很粗糙的处理，需要进一步优化
    output = np.clip(gamma_rgb, 0.0, 1.0)

    return output

def to_XYZ(src: np.ndarray, type: str):
    """
    使用 Colour 库将其它颜色空间转换为 XYZ 颜色空间。

    Parameters:
        src (np.ndarray): 输入数组，形状为 [batch, num_channels]

    Returns:
        np.ndarray: 输出数组，形状为 [batch, 3]，表示 XYZ 颜色值。
    """
    XYZ = np.zeros((src.shape[0], 3))

    if type == 'XYZ':
        XYZ = src
    elif type == 'RGB':
        # 定义 sRGB 色彩空间（D65 白点）
        sRGB = colour.models.RGB_COLOURSPACE_sRGB

        # Gamma 解码
        linear_rgb = colour.cctf_decoding(src, function="sRGB")

        # 转换为 XYZ 色彩空间
        XYZ = colour.RGB_to_XYZ(
            linear_rgb,
            colourspace=sRGB,
            illuminant_XYZ=sRGB.whitepoint,
            illuminant_RGB=sRGB.whitepoint,
            chromatic_adaptation_transform="Bradford"
        )
    elif type == 'xyY':
        x = src[:, 0]
        y = src[:, 1]
        Y = src[:, 2]
        XYZ[:, 0] = x * Y / y
        XYZ[:, 1] = Y
        XYZ[:, 2] = (1 - x - y) * Y / y
    elif type == 'BT2020':
        # 定义 BT2020 色彩空间
        BT2020 = colour.models.RGB_COLOURSPACE_BT2020

        XYZ = colour.RGB_to_XYZ(
            src,
            colourspace=BT2020,
            illuminant_XYZ=BT2020.whitepoint,
            illuminant_RGB=BT2020.whitepoint,
            chromatic_adaptation_transform="Bradford"  
        )

    return XYZ    

def BT2020_to_RGB(src: np.ndarray):
    # TODO: 实现BT2020到RGB的转换

    """
    BT2020 三原色的 xy 坐标: R(0.708,0.292), G(0.170,0.797), B(0.131,0.046)
    不妨认为 BT2020 颜色空间到 XYZ 颜色空间是线性变换 Fb ，有
    Fb(1,0,0) = [0.708 * Sr, 0.292 * Sr, 0.000 * Sr]
    Fb(0,1,0) = [0.170 * Sg, 0.797 * Sg, 0.033 * Sg]
    Fb(0,0,1) = [0.131 * Sb, 0.046 * Sb, 0.823 * Sb]
    Fb(1,1,1) = [0.9504, 1.0000, 1.0889] (D65 白点)
    不妨记作 Fb @ btRGB = XYZ （这里的 @ 是矩阵乘法，下同）
    类似地，设 sRGB 颜色空间到 XYZ 颜色空间是线性变换 Fs ，有
    Fs @ RGB = XYZ

    所以，我们可以得到 (Fs^-1 @ Fb) @ btRGB = RGB
    矩阵 M = Fs^-1 @ Fb 就是我们要求的转换矩阵。
    求出矩阵之后，剩下的问题就是处理溢出问题了。
    """

    # 当前默认是先转换为XYZ，再转换为RGB
    return XYZ2RGB(to_XYZ(src, 'BT2020'))
