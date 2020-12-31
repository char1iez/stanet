import importlib

import torch.utils.data
import torch
from torch.utils.data import ConcatDataset

from .base_dataset import BaseDataset
from .data_config import get_dataset_info


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename, __package__)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def create_single_dataset(opt, dataset_type_):
    # return dataset_class
    dataset_class = find_dataset_using_name('list')
    # get dataset root
    opt.dataroot = get_dataset_info(dataset_type_)
    return dataset_class(opt)


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        print(opt.dataset_mode)
        if opt.dataset_mode == 'concat':
            #  叠加多个数据集
            datasets = []
            #  获取concat的多个数据集列表
            self.dataset_type = opt.dataset_type.split(',')
            #  去除“,”的影响
            if self.dataset_type[-1] == '':
                self.dataset_type = self.dataset_type[:-1]
            for dataset_type_ in self.dataset_type:
                dataset_ = create_single_dataset(opt, dataset_type_)
                datasets.append(dataset_)
            self.dataset = ConcatDataset(datasets)
        else:
            dataset_class = find_dataset_using_name(opt.dataset_mode)

            self.dataset = dataset_class(opt)

        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
