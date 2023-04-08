import re
import torch


class Accumulator:
    def __init__(self, size):
        self.count = torch.ones(1, dtype=torch.float32).cuda()
        self.sum = torch.zeros(size, dtype=torch.float32).cuda()
        self.sum_squared = torch.zeros(size, dtype=torch.float32).cuda()

        self.count_reduce = None
        self.sum_reduce = None
        self.sum_squared_reduce = None

    def accumulate(self, data):
        b, n, _ = data.shape
        data = data.reshape((b * n, -1))
        sum = torch.sum(data, dim=0)
        sum_squared = torch.sum(data ** 2, dim=0)
        self.sum += sum
        self.sum_squared += sum_squared
        self.count += torch.tensor(b * n).to(self.count.device)

    def all_reduce(self):
        self.count_reduce = torch.distributed.all_reduce(self.count, async_op=True)
        self.sum_reduce = torch.distributed.all_reduce(self.sum, async_op=True)
        self.sum_squared_reduce = torch.distributed.all_reduce(self.sum_squared, async_op=True)

    def wait(self):
        self.count_reduce.wait()
        self.sum_reduce.wait()
        self.sum_squared_reduce.wait()


def unpack_filename(path):
    # search = re.search(r'.*mises_(\S+)_(\S+)_([0-9]+).npy', path)
    search = re.search(r'.*mises_(\S+)_([0-9]+).npy', path)
    if search is None:
        return '0', 0
    return search.group(1), int(search.group(2))
    #return search.group(1), search.group(2), int(search.group(3))  # shape_id, loads_id, load_index
