from scipy.io import loadmat
from simplecv.data import preprocess

from data.base import FullImageDataset
from data.base import FullImageDataset_small
SEED = 2333


class NewPaviaDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 sample_percent=0.01,
                 batch_size=10):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['paviaU']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['paviaU_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size
        super(NewPaviaDataset, self).__init__(image, mask, training, np_seed=SEED,
                                              sample_percent=sample_percent,
                                              batch_size=batch_size)

    @property
    def num_classes(self):
        return 9
class SmallPaviaDataset(FullImageDataset_small):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=10,
                 sub_minibatch=10):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['paviaU']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['paviaU_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        super(SmallPaviaDataset, self).__init__(image, mask, training, np_seed=SEED,
                                              num_train_samples_per_class=num_train_samples_per_class,
                                              sub_minibatch=sub_minibatch)

    @property
    def num_classes(self):
        return 9
