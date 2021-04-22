from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt


def convolve2d(array, filter):
    padding = math.ceil((filter.shape[0] - 1) / 2)  # padding width
    npad = ((padding, padding), (padding, padding))
    a = np.zeros((array.shape[0], array.shape[1]))
    array = np.pad(array, npad, 'constant', constant_values=(0))    # padding 적용
    filter = np.fliplr(np.flipud(filter))   # x, y축으로 뒤집음
    
    for j in range(a.shape[0]):
        for i in range(a.shape[1]):     # 각 픽셀에 대해 계산
            a[j][i] = np.sum(np.multiply(array[j:j+2*padding+1, \
                                         i:i+2*padding+1], filter))

    return a


if __name__ == '__main__':
    img = np.array(Image.open('./data/train/cat.30.jpg').convert('L'))
    sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img_filtered = np.abs(convolve2d(img, sobel_filter))

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(img_filtered, cmap='gray')
    
    plt.show()