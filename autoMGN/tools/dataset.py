import os
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import KFold


class Dataset(data.Dataset):
    """
    torch.utils.data.Dataset是一个抽象类，以字典形式存储数据，规定子类必须实现__getitem__方法，该方法接收一个key参数，返回对应的value对象
    本类封装了建图时的数据处理，针对每个汽车模型的点集合与边集合计算了初始node_feature和edge_feature
    """

    def __init__(self, config, split, mode):
        self.root = config['root']
        # 将./data/CarModel下的文件装入一个array
        self.paths = [os.path.join(self.root, filename) for filename in os.listdir(self.root)]
        self.paths = np.array(self.paths)
        size = self.paths.size
        splice = int(size / split)  # 分为几份
        if mode == 'train':
            self.paths = self.paths[splice:]
        elif mode == 'val':
            self.paths = self.paths[0:splice]

    def __getitem__(self, i):
        path = self.paths[i]
        # 文件实际序号
        index = int(os.listdir(self.root)[i][5:8])
        # 样本对应真实值（label）
        target = np.load("data/cd.npy", allow_pickle=True).item()
        target = float(target[str(index)])
        target = np.array(target)
        # print(target)
        car_model = np.load(path, allow_pickle=True)
        connections = car_model['connections']
        positions = car_model['positions']

        senders = connections[:, 0]
        receivers = connections[:, 1]
        relative_pos = positions[senders] - positions[receivers]
        edge_len = np.linalg.norm(relative_pos, axis=1, keepdims=True)
        # 点特征
        nodes = positions
        # 边特征
        edges = np.concatenate((relative_pos, edge_len), axis=1)

        senders = torch.from_numpy(senders).long()
        receivers = torch.from_numpy(receivers).long()
        nodes = torch.from_numpy(nodes).float()
        edges = torch.from_numpy(edges).float()
        target = torch.from_numpy(target).float()

        return senders, receivers, nodes, edges, target, path

    def __len__(self):
        return len(self.paths)
