import numpy as np
from simplecv.util import registry
from simplecv.interface import LearningRateBase
import math


def make_learningrate(config):
    lr_type = config['type']
    if lr_type in registry.LR:
        lr_module = registry.LR[lr_type]
        return lr_module(**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(lr_type))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@registry.LR.register('multistep')
class MultiStepLearningRate(LearningRateBase):
    def __init__(self,
                 steps,
                 base_lr=0.1,
                 gamma=0.1,
                 warmup_step=None,
                 warmup_init_lr=None):
        super(MultiStepLearningRate, self).__init__(base_lr=base_lr)
        self._steps = np.array(list(steps))
        self._gamma = gamma
        self._warmup_step = warmup_step
        self._warmup_init_lr = warmup_init_lr

        self._check()

    def _check(self):
        if self._warmup_step is not None:
            assert self._warmup_init_lr < self._base_lr
            assert self._warmup_step < self._steps[0]
        if self._steps.shape[0] > 1:
            assert np.all(np.diff(self._steps) > 0)

    def step(self, global_step, optimizer):
        cur_step = global_step

        if self._warmup_step is not None:
            if cur_step <= self._warmup_step:
                lr = self._compute_warmup_lr(cur_step)
                set_lr(optimizer, lr)
                return

        lr = self._compute_lr(cur_step)

        set_lr(optimizer, lr)

    def _compute_lr(self, cur_step):
        return self._base_lr * (self._gamma ** int((cur_step > self._steps).sum(dtype=np.int32)))

    def _compute_warmup_lr(self, cur_step):
        lr = cur_step * (self._base_lr - self._warmup_init_lr) / self._warmup_step + self._warmup_init_lr
        return lr


@registry.LR.register('poly')
class PolyLearningRate(LearningRateBase):
    def __init__(self,
                 base_lr,
                 power,
                 max_iters,
                 ):
        super(PolyLearningRate, self).__init__(base_lr)
        self.power = power
        self.max_iters = max_iters

    def step(self, global_step, optimizer):
        factor = (1 - global_step / self.max_iters) ** self.power
        cur_lr = self.base_lr * factor
        set_lr(optimizer, cur_lr)


@registry.LR.register('cosine')
class CosineAnnealingLearningRate(LearningRateBase):
    def __init__(self, base_lr, max_iters, eta_min):
        super(CosineAnnealingLearningRate, self).__init__(base_lr)
        self.eta_min = eta_min
        self.max_iters = max_iters

    def step(self, global_step, optimizer):
        cur_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * global_step / self.max_iters))

        set_lr(optimizer, cur_lr)
