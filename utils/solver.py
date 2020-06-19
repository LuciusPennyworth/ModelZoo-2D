"""
Build optimizers and schedulers

Notes:
    Default optimizer will optimize all parameters.
    Custom optimizer should be implemented and registered in '_OPTIMIZER_BUILDERS'

"""
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

_OPTIMIZER_BUILDERS = {}


def build_optimizer(cfg, model):
    name = cfg.OPTIM.TYPE
    if hasattr(torch.optim, name):
        def builder(cfg, model):
            return getattr(torch.optim, name)(
                group_weight(model, cfg.OPTIM.WEIGHT_DECAY),
                lr=cfg.OPTIM.BASE_LR,
                **cfg.OPTIM[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, model)


def group_weight(module, weight_decay):
    group_decay = []
    group_no_decay = []
    keywords = [".bn."]  # for parameters with ".bn." in their name, optimizer will not use weight_decay on them.

    for m in list(module.named_parameters()):
        exclude = False
        for k in keywords:
            if k in m[0]:
                print("Weight decay exclude: "+m[0])
                group_no_decay.append(m[1])
                exclude = True
                break
        if not exclude:
            print("Weight decay include: " + m[0])
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    # optimizer can use dict as input to make a
    groups = [dict(params=group_decay, weight_decay=weight_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if hasattr(torch.optim.lr_scheduler, name):
        def builder(cfg, optimizer):
            return getattr(torch.optim.lr_scheduler, name)(
                optimizer,
                **cfg.SCHEDULER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, optimizer)
