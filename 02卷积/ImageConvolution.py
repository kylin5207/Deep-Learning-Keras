# -- coding = 'utf-8' -- 
# Author Kylin
# Python Version 3.7.3
# OS macOS
"""
图像卷积
"""
import numpy as np
import os
from PIL import Image


def convolve(image, weight):
    """
    对图片进行简单的卷积处理
    :param image:  图片数据
    :param weight: 权重数据
    :return:卷积结果
    """
    # 1. 图片大小与卷积核大小
    height, width = image.shape
    h, w = weight.shape

    # 2. 卷积结果大小
    height_new = height - h + 1
    width_new = width - w + 1

    # 3. 新数据
    image_new = np.empty((height_new, width_new), dtype=np.float)

    # 4. 卷积操作
    for i in range(height_new):
        for j in range(width_new):
            image_new[i,j] = np.sum(image[i:i+h, j:j+w] * weight)

    # 5. 截取在[0,255]间的数据
    image_new = image_new.clip(0, 255)

    # 6. 四舍五入取整数
    image_new = np.rint(image_new).astype('uint8')
    return image_new

# image_new = 255 * (image_new - image_new.min()) / (image_new.max() - image_new.min())


def main():
    # 1. 读取图像
    A = Image.open("./pic/timo.jpeg", 'r')
    print("图像数据：")
    print(A)  # emm,是一个对象

    # 1.2 卷积后存放图像的路径
    output_path = './pic/ImageConvolve/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 2. 图像预处理
    # 便于处理，转为数组
    a = np.array(A)
    print("图像数组大小：", a.shape) # 三通道数据
    print(a) # R G B 三个通道的数据

    # 3. 卷积核数据
    # 3 * 3列卷积核
    avg3 = np.ones((3, 3))
    avg3 /= avg3.sum()
    print(avg3)

    # 5 * 5列卷积核
    avg5 = np.ones((5, 5))
    avg5 /= avg5.sum()

    # 20 * 20列卷积核
    avg20 = np.ones((20, 20))
    avg20 /= avg20.sum()

    # 高斯核
    gauss = np.array(([0.003, 0.013, 0.022, 0.013, 0.003],
                      [0.013, 0.059, 0.097, 0.059, 0.013],
                      [0.022, 0.097, 0.159, 0.097, 0.022],
                      [0.013, 0.059, 0.097, 0.059, 0.013],
                      [0.003, 0.013, 0.022, 0.013, 0.003]))

    # 各种核
    soble_x = np.array(([-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]))
    soble_y = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    soble = np.array(([-1, -1, 0],
                      [-1, 0, 1],
                      [0, 1, 1]))
    prewitt_x = np.array(([-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 2]))
    laplacian4 = np.array(([0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]))
    laplacian8 = np.array(([-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]))
    weight_list = (
    'avg3', 'avg5', 'avg20', 'gauss', 'soble_x', 'soble_y', 'soble', 'prewitt_x', 'prewitt_y', 'prewitt', 'laplacian4',
    'laplacian8')

    print('梯度检测：')
    for weight in weight_list:
        # 分别对RGB三通道进行卷积
        print(weight, 'R', end=' ')
        R = convolve(a[:, :, 0], eval(weight))

        print('G', end=' ')
        G = convolve(a[:, :, 1], eval(weight))

        print('B')
        B = convolve(a[:, :, 2], eval(weight))

        # 合成图像数据
        I = np.stack((R, G, B), 2)
        if weight not in ('avg3', 'avg5', 'avg20', 'gauss'):
            I = 255 - I

        # 4. 保存数据
        Image.fromarray(I).save(output_path + weight + '.png')


if __name__ == "__main__":
    main()