from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像
#gray_image = io.imread('../testImage/NoColor.png', as_gray=True)


# 定义颜色映射
def grey2Color(image):
    # 将灰度值映射到RGB颜色空间
    r = np.zeros_like(image)
    g = np.zeros_like(image)
    b = np.zeros_like(image)

    # 示例：简单的颜色映射
    # 小于0

    r[np.where((image < 0))] = 0
    g[np.where((image < 0))] = 174
    b[np.where((image < 0))] = 165

    # 0-5
    r[np.where((image >= 0) & (image < 5))] = 198
    g[np.where((image >= 0) & (image < 5))] = 195
    b[np.where((image >= 0) & (image < 5))] = 255

    # 5-10
    r[np.where((image >= 5) & (image < 10))] = 123
    g[np.where((image >= 5) & (image < 10))] = 113
    b[np.where((image >= 5) & (image < 10))] = 239

    # 10-15
    r[np.where((image >= 10) & (image < 15))] = 24
    g[np.where((image >= 10) & (image < 15))] = 36
    b[np.where((image >= 10) & (image < 15))] = 214

    # 15-20
    r[np.where((image >= 15) & (image < 20))] = 165
    g[np.where((image >= 15) & (image < 20))] = 255
    b[np.where((image >= 15) & (image < 20))] = 173

    # 20-25
    r[np.where((image >= 20) & (image < 25))] = 0
    g[np.where((image >= 20) & (image < 25))] = 235
    b[np.where((image >= 20) & (image < 25))] = 0

    # 25-30
    r[np.where((image >= 25) & (image < 30))] = 16
    g[np.where((image >= 25) & (image < 30))] = 146
    b[np.where((image >= 25) & (image < 30))] = 24

    # 30-35
    r[np.where((image >= 30) & (image < 35))] = 255
    g[np.where((image >= 30) & (image < 35))] = 247
    b[np.where((image >= 30) & (image < 35))] = 99

    # 35-40
    r[np.where((image >= 35) & (image < 40))] = 206
    g[np.where((image >= 35) & (image < 40))] = 203
    b[np.where((image >= 35) & (image < 40))] = 0

    # 40-45
    r[np.where((image >= 40) & (image < 45))] = 140
    g[np.where((image >= 40) & (image < 45))] = 142
    b[np.where((image >= 40) & (image < 45))] = 0

    # 45-50
    r[np.where((image >= 45) & (image < 50))] = 255
    g[np.where((image >= 45) & (image < 50))] = 174
    b[np.where((image >= 45) & (image < 50))] = 173

    # 50-55
    r[np.where((image >= 50) & (image < 55))] = 255
    g[np.where((image >= 50) & (image < 55))] = 101
    b[np.where((image >= 50) & (image < 55))] = 82

    # 55-60
    r[np.where((image >= 55) & (image < 60))] = 239
    g[np.where((image >= 55) & (image < 60))] = 0
    b[np.where((image >= 55) & (image < 60))] = 49

    # 60-65
    r[np.where((image >= 60) & (image < 65))] = 214
    g[np.where((image >= 60) & (image < 65))] = 142
    b[np.where((image >= 60) & (image < 65))] = 255

    # 65-
    r[np.where((image >= 65))] = 173
    g[np.where((image >= 65))] = 36
    b[np.where((image >= 65))] = 255

    return np.stack((r, g, b), axis=-1)

#max_value = np.max(gray_image)

#print("二维数组中的最大值是:", max_value)

# 应用伪彩色
# color_image = pseudocolor(gray_image)
#
# colored_image_uint8 = (color_image).astype(np.uint8)
# # 显示彩色图像
# plt.imshow(colored_image_uint8)
# plt.show()
#
# # 保存彩色图像
# io.imsave('../resImage/Color.png', colored_image_uint8)