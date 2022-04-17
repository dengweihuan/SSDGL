import math
import torch
from torch.utils.data.distributed import DistributedSampler


class StepDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        super(StepDistributedSampler, self).__init__(dataset, num_replicas, rank)
        self.step = 0

    def set_step(self, step):
        self.step = step

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.step)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class StepDistributedRandomSubsetSampler(StepDistributedSampler):
    def __init__(self, indices, num_replicas=None, rank=None):
        super(StepDistributedRandomSubsetSampler, self).__init__([], num_replicas, rank)

        self.indices = indices
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.step)
        indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_step(self, step):
        self.step = step
