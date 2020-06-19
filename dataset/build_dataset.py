"""
Created on 2020/6/11 18:41 周四
@author: Matt zhuhan1401@126.com
Description: description
"""

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms


def build_data_loader(cfg, name='MNIST', mode='train'):
    if name == 'MNIST':
        if mode == 'train':
            dataset = datasets.MNIST(root=r'F:\PythonCode\data\MNIST',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
        else:
            dataset = datasets.MNIST(root=r'F:\PythonCode\data\MNIST',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=False)

    elif name == 'CIFAR10':
        if mode == 'train':
            dataset = datasets.CIFAR10(root=r'F:\PythonCode\data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
        else:
            dataset = datasets.CIFAR10(root=r'F:\PythonCode\data',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=False)

    batch_size = cfg.GLOBAL.TRAIN.BATCH_SIZE if mode == 'train' else cfg.GLOBAL.TEST.BATCH_SIZE

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False) # mode=='train'

    return data_loader


if __name__ == '__main__':
    import argparse
    from utils import config

    def get_parser():
        parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
        parser.add_argument('--config_file', type=str,
                            default='F:\PythonCode\ModelZoo-2D\configs\default.yaml', help='config file')
        parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb_win.yaml for all options', default=None,
                            nargs=argparse.REMAINDER)
        args = parser.parse_args()
        assert args.config_file is not None
        cfg = config.load_cfg_from_file(args.config_file)
        if args.opts is not None:
            cfg.merge_from_list(args.opts)
        return cfg

