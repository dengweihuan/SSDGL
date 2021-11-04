from torch.utils.data import dataset
import numpy as np
from simplecv.data.preprocess import divisible_pad
import torch
from torch.utils import data
from random import shuffle
SEED = 2333

class FullImageDataset_hos(dataset.Dataset):
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2333,
                 num_train_samples_per_class=10,
                 sub_minibatch=10,
                 ):
        self.image = image
        self.mask = mask
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def preset(self):
        if self.training:

            #print(self.num_train_samples_per_class)

            train_indicator, test_indicator = fixed_num_sample_hos(self.mask, self.num_train_samples_per_class,self.num_classes, self._seed)

            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16, False)
            im = blob[0, :self.image.shape[-1], :, :]

            mask = blob[0, -3, :, :]
            self.train_indicator = blob[0, -2, :, :]
            self.test_indicator = blob[0, -1, :, :]


            self.train_inds_list = minibatch_sample_hos(mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

            self.pad_im = im
            self.pad_mask = mask
            np.save('train_indicator.npy', self.train_indicator)
            np.save('test_indicator.npy', self.test_indicator)
            np.save('pad_im.npy', self.pad_im)
            np.save('pad_mask.npy', self.pad_mask)
        else:
            self.train_indicator,self.test_indicator =np.load('train_indicator.npy'), np.load('test_indicator.npy')
            self.pad_im = np.load('pad_im.npy')
            self.pad_mask = np.load('pad_mask.npy')


    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample_hos(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 15

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1

class FullImageDataset(dataset.Dataset):
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2333,
                 sample_percent=0.01,
                 batch_size=10,
                 ):
        self.image = image
        self.mask = mask
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)

        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def preset(self):
        if self.training:

            #print(self.num_train_samples_per_class)

            train_indicator, test_indicator = fixed_num_sample(self.mask, self.sample_percent,self.num_classes, self._seed)

            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16, False)
            im = blob[0, :self.image.shape[-1], :, :]

            mask = blob[0, -3, :, :]
            self.train_indicator = blob[0, -2, :, :]
            self.test_indicator = blob[0, -1, :, :]


            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.batch_size,
                                                    seed=self.seeds_for_minibatchsample.pop())

            self.pad_im = im
            self.pad_mask = mask
            np.save('train_indicator.npy', self.train_indicator)
            np.save('test_indicator.npy', self.test_indicator)
            np.save('pad_im.npy', self.pad_im)
            np.save('pad_mask.npy', self.pad_mask)
        else:
            self.train_indicator,self.test_indicator =np.load('train_indicator.npy'), np.load('test_indicator.npy')
            self.pad_im = np.load('pad_im.npy')
            self.pad_mask = np.load('pad_mask.npy')


    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.batch_size,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 16

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1

class MinibatchSampler_hos(data.Sampler):
    def __init__(self, dataset: FullImageDataset_hos):
        super(MinibatchSampler_hos, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)


class MinibatchSampler(data.Sampler):
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)

def fixed_num_sample_hos(gt_mask: np.ndarray, num_train_samples, num_classes, seed=2333):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)
    shuchu=[]
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        shuffle(inds)
        #print("num_train_samples",num_train_samples)
        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]
        shuchu.extend(train_inds)
        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1
        #print("shuchu",shuchu)
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)

    return train_indicator, test_indicator

def fixed_num_sample(gt_mask: np.ndarray, sample_percent, num_classes, seed=2333):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)
    #(106，610，340)
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    shuchu=[]

    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        count=np.sum(gt_mask_flatten == i)
        num_train_samples=np.ceil(count*sample_percent)
        num_train_samples = num_train_samples.astype(np.int32)
        if num_train_samples <5:
            num_train_samples=5 # At least 5 samples per class
        shuffle(inds)

        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]
        shuchu.extend(train_inds)

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1
        #print("shuchu",shuchu)
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    return train_indicator, test_indicator

def minibatch_sample_hos(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    """
    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:

    Returns:
    """
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            shuffle(inds)
            cd = len(inds)
            fetch_inds = inds[:cd]
            train_inds[fetch_inds] = 1
        cnt += 1
        if cnt == 11:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))


def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, batch_size, seed):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:

    Returns:

    """
    rs = np.random.RandomState(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
                shuffle(inds)
                cd=min(batch_size, len(inds))
                fetch_inds = inds[:cd]
                train_inds[fetch_inds] = 1

        cnt += 1
        if cnt == 11:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))

