import os
import yaml


def load_train_config():
    with open(os.path.join('config', 'train.yaml'), 'r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)


def load_test_config():
    with open(os.path.join('config', 'test.yaml'), 'r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)
