from torch import nn
from model.components import EncodeProcessDecode, Normalizer


class MGN(nn.Module):
    """
    相当于在执行EncodeProcessDecode之前添加了Normalizer层
    """
    def __init__(self, config):
        super(MGN, self).__init__()
        self.config = config

        self.node_normalizer = Normalizer(size=config['node_feat_size'])
        self.edge_normalizer = Normalizer(size=config['edge_feat_size'])
        self.output_normalizer = Normalizer(size=config['output_feat_size'])

        self.epd_component = EncodeProcessDecode(input_dim_node=config['node_feat_size'],
                                                 input_dim_edge=config['edge_feat_size'],
                                                 hidden_dim=config['latent_size'],
                                                 output_dim=config['output_feat_size'],
                                                 num_layers=config['num_layers'],
                                                 message_passing_steps=config['message_passing_steps'])

    def accumulate(self, node_features, edge_features, targets):
        self.node_normalizer.accumulate(node_features)
        self.edge_normalizer.accumulate(edge_features)
        self.output_normalizer.accumulate(targets)

    def output_normalize(self, data):
        return self.output_normalizer(data)

    def output_normalize_inverse(self, data):
        return self.output_normalizer.inverse(data)

    def forward(self, senders, receivers, node_features, edge_features, is_connected):
        node_features = self.node_normalizer(node_features)
        edge_features = self.edge_normalizer(edge_features)
        prediction = self.epd_component(senders, receivers, node_features, edge_features, is_connected)
        return prediction

