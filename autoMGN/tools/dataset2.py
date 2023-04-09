import os
import random
import numpy as np
import torch
import torch.utils.data as data
from .common import unpack_filename


class Dataset(data.Dataset):
    """
    torch.utils.data.Dataset是一个抽象类，以字典形式存储数据，规定子类必须实现__getitem__方法，该方法接收一个key参数，返回对应的value对象
    本类封装了建图时的数据处理，针对每个汽车模型的点集合与边集合计算了初始node_feature和edge_feature
    """
    def __init__(self, config, parts=None, npart=10, ids=None, shuffle=True):
        self.root = config['root']
        # 将./data/CarModel下的文件装入一个array
        self.paths = [os.path.join(self.root, filename) for filename in
                      os.listdir(self.root)]
        self.paths = np.array(self.paths)

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
        # path = self.paths[i]
        #
        # car_model = np.load(path)
        # # [['connections'] ['positions']]
        # car_model = np.expand_dims(car_model, axis=1)
        #
        # #shape_id, loads_id, load_index = unpack_filename(path)
        # load_index, shape_id = unpack_filename(path)
        # #shape = np.load(os.path.join(self.root, 'shapes', 'shape_%s.npz' % shape_id))
        # shape = np.load(os.path.join(self.root, 'Model%s.npz' % shape_id))
        # connections = shape['connections']
        # positions = shape['positions'], 'loads_%s_%s.npy' % (shape_id, loads_id)[load_index]
        # load = np.tile(load, (len(positions), 1))
        #
        # senders = connections[:, 0]
        # receivers = connections[:, 1]
        # relative_pos = positions[senders] - positions[receivers]
        # edge_len = np.linalg.norm(relative_pos, axis=1, keepdims=True)
        #
        # nodes = load
        # edges = np.concatenate((relative_pos, edge_len), axis=1)
        #
        # senders = torch.from_numpy(senders).long()
        # receivers = torch.from_numpy(receivers).long()
        # nodes = torch.from_numpy(nodes).float()
        # edges = torch.from_numpy(edges).float()
        # car_model = torch.from_numpy(car_model).float()
        #
        # return senders, receivers, nodes, edges, car_model, path
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


        #load = np.load(os.path.join(self.root, 'loads')