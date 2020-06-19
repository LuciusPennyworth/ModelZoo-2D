"""
Created on 2020/6/11 17:29 周四
@author: Matt zhuhan1401@126.com
Description: description
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.layer2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.layer1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        out = self.layer2(x)
        return out


class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()
        self.crossEntropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        losses = {}
        losses['crossEntry'] = self.crossEntropy(pred, target)
        return losses


class NetMetric(nn.Module):

    def __init__(self, cfg, mode):
        super(NetMetric, self).__init__()
        self.cfg = cfg
        self.mode = mode

    def forward(self, pred, input, target):
        metrics = {}

        metrics['correct_num'] = torch.sum(torch.argmax(pred.data, dim=1) == target.data).item()
        batch_size = self.cfg.GLOBAL.TRAIN.BATCH_SIZE if self.mode else cfg.GLOBAL.TEST.BATCH_SIZE
        metrics['acc'] = metrics['correct_num'] / batch_size

        return metrics

    def visu_classify_result(self, pred, input, target):
        zero_ele_idx = torch.nonzero((torch.argmax(pred.data, dim=1) == target.data) == 0)
        if len(zero_ele_idx) != 0:
            pred = pred.cpu()
            input = input.cpu()
            target = target.cpu()
            idx = zero_ele_idx[0]
            input = input[idx, 0, :, :].data.squeeze()
            target = target[idx]
            pred = torch.argmax(pred[idx])

            plt.imshow(input)
            plt.title("targe:{} pred:{}".format(target.data, pred))
            plt.show()


def build_model(cfg, mode='train'):
    model = cfg.GLOBAL.MODEL
    if model == 'ResNet18':
        from model.ResNet import resnet18
        model = resnet18(cfg.GLOBAL.CLASSES)
    else:
        model = Net(cfg.GLOBAL.INPUT_SIZE, cfg.GLOBAL.CLASSES)
    loss_fn = NetLoss()
    metric_fn = NetMetric(cfg, mode)
    return model, loss_fn, metric_fn


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

    cfg = get_parser()
    build_model(cfg)