import os
import random
import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict
from sklearn.cluster import KMeans

class Dataset(data.Dataset):
    """
    torch.utils.data.Dataset是一个抽象类，以字典形式存储数据，规定子类必须实现__getitem__方法，该方法接收一个key参数，返回对应的value对象
    本类封装了建图时的数据处理，针对每个汽车模型的点集合与边集合计算了初始node_feature和edge_feature
    """
    def __init__(self, config, parts=None, npart=10, ids=None, shuffle=True):
        self.root = config['root']
        # 将./data/CarModel下的文件装入一个array
        self.paths = [os.path.join(self.root, filename) for filename in os.listdir(self.root) if
                      filename.startswith("Model")]
        self.paths = np.array(self.paths)
        self.num_clusters = 10  # Number of clusters
        # if ids is not None:
        #     ids = set(ids)   # 转为一个set
        #     self.paths = filter(lambda path: unpack_filename(path)[0] in ids, self.paths)
        #     # 第一个为过滤条件，第二个为iterable容器，返回所有判定为True的元素构成的容器
        #     self.paths = np.array(list(self.paths))

        if parts is not None:
            n = self.paths.size
            d = int(n / npart)
            index = np.arange(0, n)
            if shuffle:
                random.shuffle(index)
            index = index[np.array([np.arange(part * d, (part + 1) * d) for part in parts]).flatten()]
            self.paths = self.paths[index]
            print(index)

        print(self.paths.size)
        print(self.paths)

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
        winds = car_model['winds'].reshape(-1, 1)

        senders = connections[:, 0]
        receivers = connections[:, 1]

        relative_pos = positions[senders] - positions[receivers]
        edge_len = np.linalg.norm(relative_pos, axis=1, keepdims=True)
        # 点特征
        nodes = positions
        # 边特征
        edges = np.concatenate((relative_pos, edge_len, winds), axis=1)

        # 使用 K-Means 对节点进行聚类
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10)
        node_labels = kmeans.fit_predict(nodes)

        senders = torch.from_numpy(senders).long()
        receivers = torch.from_numpy(receivers).long()
        nodes = torch.from_numpy(nodes).float()
        edges = torch.from_numpy(edges).float()
        target = torch.from_numpy(target).float()
        node_labels = torch.from_numpy(node_labels).long()

        is_connected = []
        for i, (start, end) in enumerate(zip(senders, receivers)):
            # 不同聚类的节点不能交流
            if node_labels[start] != node_labels[end]:
                # print("不同聚类：%f" % i)
                is_connected.append(0)
            else:
                is_connected.append(1)

        # 将 is_connected 转换为一维数组
        is_connected = np.array(is_connected)

        edges[is_connected == 0, :] = 0
        is_connected = torch.from_numpy(is_connected).long()
        mask = is_connected.unsqueeze(-1).bool()
        # print("mask: ", mask.shape)
        return senders, receivers, nodes, edges, target, path, mask
    def __len__(self):
        return len(self.paths)


        #load = np.load(os.path.join(self.root, 'loads')