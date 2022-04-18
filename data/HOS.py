from scipy.io import loadmat
from simplecv.data import preprocess
import torch.nn as nn
from data.base import FullImageDataset_small
import torch
import torch.nn as nn
SEED = 2333


class NewHOSDataset(FullImageDataset_small):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=10,
                 sub_minibatch=10):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['ans']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['name']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)

        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch

        super(NewHOSDataset, self).__init__(image, mask, training, np_seed=SEED,
                                                    num_train_samples_per_class=num_train_samples_per_class,
                                                    sub_minibatch=sub_minibatch)

