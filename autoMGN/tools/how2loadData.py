import numpy as np

if __name__ == '__main__':
    """
    读取npy文件
    """
    # Cd为风阻值 Coefficient of drag
    Cd = np.load("../data/cd.npy", allow_pickle=True)
    print(Cd)  # {'1': '0.357827734', '2': '0.3951254', ... }

    """
    读取npz文件
    """
    model = np.load("../data/CarModel/Model001.npz", allow_pickle=True)
    print(model.files)  # ['connections', 'positions']，边和点
    print(np.array(model['connections']).shape)  # (103872, 2)，边数和边的两个顶点
    print(np.array(model['positions']).shape)  # (14844, 3)，点数和点的三维坐标
