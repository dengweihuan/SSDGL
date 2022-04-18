from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from data.pavia import NewPaviaDataset
from data.pavia import SmallPaviaDataset
from data.base import MinibatchSampler
from data.salinas import NewSalinasDataset
from data.indianpine import NewIndianPinesDataset
from data.indianpine import SmallIndianPinesDataset
from data.HOS import NewHOSDataset

@registry.DATALOADER.register('SmallIndianPinesLoader')
class SmallIndianPinesLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = SmallIndianPinesDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch)
        print("gf",self.num_train_samples_per_class)
        sampler = MinibatchSampler(dataset)
        super(SmallIndianPinesLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=10,
            # mini-batch per class, if there are 9 categories, the total mini-batch is sub_minibatch * num_classes (9)
            sub_minibatch=10
        ))
@registry.DATALOADER.register('SmallPaviaLoader')
class SmallPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = SmallPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.num_train_samples_per_class, self.sub_minibatch)
        sampler = MinibatchSampler(dataset)
        super(SmallPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=10,
            sub_minibatch=10
        ))

@registry.DATALOADER.register('NewPaviaLoader')
class NewPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.sample_percent, self.batch_size)
        sampler = MinibatchSampler(dataset)
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )


@registry.DATALOADER.register('NewSalinasLoader')
class NewSalinasLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size)
        sampler = MinibatchSampler(dataset)
        super(NewSalinasLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )

@registry.DATALOADER.register('NewIndianPinesLoader')
class NewIndianPinesLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewIndianPinesDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size)
        sampler = MinibatchSampler(dataset)
        super(NewIndianPinesLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )
@registry.DATALOADER.register('NewHOSLoader')
class NewHOSLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHOSDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.num_train_samples_per_class, self.sub_minibatch)
        sampler = MinibatchSampler(dataset)
        super(NewHOSLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            num_train_samples_per_class=10,
            # mini-batch per class, if there are 10 categories, the total mini-batch is sub_minibatch * num_classes (10)
            sub_minibatch=10
        ))

