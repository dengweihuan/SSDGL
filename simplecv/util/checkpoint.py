import os
from collections import OrderedDict
import torch
import json
from simplecv.util.logger import get_logger, save_log, restore_log

logger = get_logger(__name__)


def is_checkpoint(obj):
    if isinstance(obj, CheckPoint):
        return True
    if isinstance(obj, OrderedDict) and all([
        CheckPoint.MODEL in obj,
        CheckPoint.OPTIMIZER in obj,
        CheckPoint.GLOBALSTEP in obj
    ]):
        return True
    return False


class CheckPoint(object):
    MODEL = 'model'
    OPTIMIZER = 'opt'
    GLOBALSTEP = 'global_step'
    LASTCHECKPOINT = 'last'
    CHECKPOINT_NAME = 'checkpoint_info.json'

    def __init__(self, launcher=None):
        self._launcher = launcher
        self._global_step = 0
        self._json_log = {CheckPoint.LASTCHECKPOINT: dict(
            step=0,
            name=''),
        }
        self.init_checkpoint_info_from_launcher()

    def set_global_step(self, value):
        if value >= 0:
            self._global_step = value
        else:
            raise ValueError('The global step must be larger than zero.')

    @property
    def global_step(self):
        return self._global_step

    def step(self):
        self._global_step += 1

    def set_launcher(self, launcher):
        self._launcher = launcher
        self.init_checkpoint_info_from_launcher()

    def save(self):
        ckpt = OrderedDict({
            CheckPoint.MODEL: self._launcher.model.state_dict(),
            CheckPoint.OPTIMIZER: self._launcher.optimizer.state_dict(),
            CheckPoint.GLOBALSTEP: self.global_step
        })
        filename = self.get_checkpoint_name(self.global_step)
        filepath = os.path.join(self._launcher.model_dir, filename)
        torch.save(ckpt, filepath)
        self._json_log[self.global_step] = filename
        if self.global_step > self._json_log[CheckPoint.LASTCHECKPOINT]['step']:
            self._json_log[CheckPoint.LASTCHECKPOINT]['step'] = self.global_step
            self._json_log[CheckPoint.LASTCHECKPOINT]['name'] = filename
        self.save_checkpoint_info(self._launcher.model_dir)
        # log
        save_log(logger, filename)

    @staticmethod
    def load(filepath):
        ckpt = torch.load(filepath, map_location=torch.device("cpu"))

        return ckpt

    def save_checkpoint_info(self, model_dir):
        with open(os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME), 'w') as f:
            json.dump(self._json_log, f)
        save_log(logger, CheckPoint.CHECKPOINT_NAME)

    def try_resume(self):
        """ json -> ckpt_path -> ckpt -> launcher

        Returns:

        """
        if self._launcher is None:
            return
        # 1. json
        model_dir = self._launcher.model_dir
        json_log = self.load_checkpoint_info(model_dir)
        if json_log is None:
            return
        # 2. ckpt path
        last_path = os.path.join(self._launcher.model_dir, json_log[CheckPoint.LASTCHECKPOINT]['name'])
        # 3. ckpt
        ckpt = self.load(last_path)
        # 4. resume

        self._launcher.model.load_state_dict(ckpt[CheckPoint.MODEL])
        if self._launcher.optimizer is not None:
            self._launcher.optimizer.load_state_dict(ckpt[CheckPoint.OPTIMIZER])
        if self._launcher.checkpoint is not None:
            self._launcher.checkpoint.set_global_step(ckpt[CheckPoint.GLOBALSTEP])
        # log
        restore_log(logger, last_path)

    def init_checkpoint_info_from_launcher(self):
        if self._launcher is None:
            return

        model_dir = self._launcher.model_dir
        json_file = self.load_checkpoint_info(model_dir)

        if json_file is None:
            return
        self._json_log = json_file

    @staticmethod
    def load_checkpoint_info(model_dir):
        json_path = os.path.join(model_dir, CheckPoint.CHECKPOINT_NAME)
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file

    @staticmethod
    def get_checkpoint_name(global_step):
        return 'model-{}.pth'.format(global_step)


def load_model_state_dict_from_ckpt(filepath):
    try:
        ckpt = torch.load(filepath)
    except RuntimeError:
        ckpt = torch.load(filepath, map_location=lambda storage, loc: storage)
    statedict = ckpt[CheckPoint.MODEL]
    ret = {}
    for k, v in statedict.items():
        k = k.replace('module.', '')
        ret[k] = v

    return ret
