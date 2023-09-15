if __name__ == '__main__':
    import numpy as np

    # 创建一个15x5的网格
    grid = np.zeros((15, 5))

    # 创建一个75x75的掩码
    mask = np.zeros((75, 75))

    # 遍历掩码中的每个单元格
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # 计算网格坐标
            grid_i = i // 5
            grid_j = j // 15

            # 检查当前单元格是否是临近的网格单元格或中心单元格
            if i % 5 == 0 or j % 15 == 0 or (grid_i, grid_j) == (2, 7):
                mask[i, j] = 1.0

    print(mask)
    #画出掩码
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.show()