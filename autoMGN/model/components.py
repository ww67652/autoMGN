import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add
import numpy as np
from collections import defaultdict
import random

class MLP(nn.Module):
    """
    基本组件：多层感知机
    1.encode阶段对所有node和edge的feature通过mlp处理
    2.process阶段每步通过mlp对所有nodes,edges的feature进行更新
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 layer_norm=True,
                 activation=nn.ReLU(),
                 activate_final=False):

        super(MLP, self).__init__()
        # input layer
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        # hidden layers
        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        # output layer
        if activate_final:
            layers += [nn.Linear(hidden_dim, output_dim), activation]
        else:
            layers += [nn.Linear(hidden_dim, output_dim)]
        # norm layer
        if layer_norm:
            layers += [nn.LayerNorm(output_dim)]

        self.net = nn.Sequential(*layers)

    # torch.nn.Module 的__call__(self)函数中会返回 forward()函数 的结果，因此PyTorch中的forward()函数可以直接通过类名被调用，而不用实例化对象
    def forward(self, input):
        output = self.net(input)

        return output


class GraphNetBlock(nn.Module):
    """
    用来执行process步骤的组件（更新所有nodes,edges的feature），我们知道process每次更新是通过mlp完成的，构造器传入的参数规定了这个mlp是怎样的
    """
    def __init__(self, hidden_dim, num_layers):
        super(GraphNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp_node = MLP(input_dim=2 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers)  # 3*hidden_dim: [nodes, accumulated_edges]
        self.mlp_edge = MLP(input_dim=3 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers)  # 3*hidden_dim: [sender, edge, receiver]

    def update_edges(self, senders, receivers, node_features, edge_features):
        senders = senders.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        receivers = receivers.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        sender_features = torch.gather(node_features, dim=1, index=senders)
        receiver_features = torch.gather(receivers, dim=1, index=receivers)
        features = torch.cat([sender_features, receiver_features, edge_features], dim=-1)

        return self.mlp_edge(features)

    def update_nodes(self, receivers, node_features, edge_features, node_sample=10):
        degrees = scatter_add(torch.ones_like(receivers), receivers, dim=0)  # 计算每个节点的度数
        mask = degrees >= 6  # 度数大于等于6的节点作为重要节点进行采样

        # 重要节点的度重要性采样
        important_nodes = torch.nonzero(mask).squeeze(1)
        important_receivers = receivers[important_nodes]
        important_edge_features = edge_features[important_receivers]
        important_accumulate_edges = scatter_add(important_edge_features, important_receivers, dim=0)

        # 非重要节点的普通采样
        non_important_nodes = torch.nonzero(~mask).squeeze(1)
        non_important_receivers = receivers[non_important_nodes]
        non_important_edge_features = edge_features[non_important_receivers]
        non_important_accumulate_edges = scatter_add(non_important_edge_features, non_important_receivers, dim=0)

        # 将重要节点和非重要节点的采样结果合并
        accumulate_edges = torch.zeros_like(edge_features)
        accumulate_edges[important_receivers] = important_accumulate_edges
        accumulate_edges[non_important_receivers] = non_important_accumulate_edges

        # 拼接节点特征和邻居特征，输入到节点更新层中
        features = torch.cat([node_features, accumulate_edges], dim=-1)
        return self.mlp_node(features)

    # def update_nodes(self, receivers, node_features, edge_features, node_sample=3):
    #     adj_list = [torch.where(receivers == index)[0] for index in range(node_features.shape[1])]
    #     node_sample_list = [min(len(adj), node_sample) for adj in adj_list]
    #
    #     # 构建空的目标张量
    #     # batch_size × node_nums × edge_features
    #     target_size = (edge_features.shape[0], node_features.shape[1], edge_features.shape[2])
    #     accumulate_edges = torch.zeros(target_size, device='cuda')
    #
    #     # 批量计算邻居特征和求和结果
    #     adj_samples = []
    #     for i, adj in enumerate(adj_list):
    #         if len(adj) > node_sample_list[i]:
    #             adj_samples.append(np.random.choice(adj.cpu().numpy(), size=node_sample_list[i], replace=False))
    #         else:
    #             adj_samples.append(adj.cpu().numpy())
    #     adj_samples = torch.from_numpy(np.stack(adj_samples)).cuda()
    #
    #     feature_sum = torch.sum(edge_features[:, adj_samples, :], dim=2)
    #     accumulate_edges[:, torch.arange(node_features.shape[1]), :] = feature_sum
    #
    #     # 拼接节点特征和邻居特征，输入到节点更新层中
    #     features = torch.cat([node_features, accumulate_edges], dim=-1)
    #     print("update")
    #     return self.mlp_node(features)

    def forward(self, senders, receivers, node_features, edge_features):
        new_edge_features = self.update_edges(senders, receivers, node_features, edge_features)
        new_node_features = self.update_nodes(receivers, node_features, new_edge_features)

        new_node_features += node_features
        new_edge_features += edge_features

        return new_node_features, new_edge_features


class Encoder(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.node_mlp = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activate_final=False)
        self.edge_mlp = MLP(input_dim=input_dim_edge, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activate_final=False)

    def forward(self, node_features, edge_features):
        node_latents = self.node_mlp(node_features)
        edge_latents = self.edge_mlp(edge_features)

        return node_latents, edge_latents


class Process(nn.Module):
    # 每次更新调用GraphNetBlock.forward(),执行 message_passing_steps 次更新
    def __init__(self, hidden_dim, num_layers, message_passing_steps):
        super(Process, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(message_passing_steps):
            self.blocks.append(GraphNetBlock(hidden_dim, num_layers))

    def forward(self, senders, receivers, node_features, edge_features):
        for graphnetblock in self.blocks:
            node_features, edge_features = graphnetblock(senders, receivers, node_features, edge_features)

        return node_features, edge_features


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       layer_norm=False, activate_final=False)

    def forward(self, node_features):
        return self.mlp(node_features)


class EncodeProcessDecode(nn.Module):
    """
    核心组件，封装了完整的EPD步骤
    """
    def __init__(self,
                 input_dim_node,
                 input_dim_edge,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 message_passing_steps):
        super(EncodeProcessDecode, self).__init__()

        self.encoder = Encoder(input_dim_node, input_dim_edge, hidden_dim, num_layers)
        self.process = Process(hidden_dim, num_layers, message_passing_steps)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)

    def forward(self, senders, receivers, node_features, edge_features):
        node_features, edge_features = self.encoder(node_features, edge_features)
        node_features, edge_features = self.process(senders, receivers, node_features, edge_features)
        predict = self.decoder(node_features)
        # predict_mean = torch.mean(predict, dim=1, keepdim=True)
        return predict


class Normalizer(nn.Module):
    def __init__(self, size, std_epsilon=1e-8):
        super(Normalizer, self).__init__()

        self.register_buffer('std_epsilon', torch.tensor(std_epsilon))
        self.register_buffer('count', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('sum', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('sum_squared', torch.zeros(size, dtype=torch.float32))

    def set_accumulated(self, accumulator):
        self.count = accumulator.count
        self.sum = accumulator.sum
        self.sum_squared = accumulator.sum_squared

    def forward(self, data):
        return (data - self.mean()) / self.std()

    def inverse(self, normalized_data):
        return normalized_data * self.std() + self.mean()

    def mean(self):
        return self.sum / self.count

    def std(self):
        std = torch.sqrt(self.sum_squared / self.count - self.mean() ** 2)
        return torch.maximum(std, self.std_epsilon)
