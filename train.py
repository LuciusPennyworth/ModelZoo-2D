"""
Created on 2020/6/10 11:46 周三
@author: Matt zhuhan1401@126.com
Description: description
"""

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import logging
import numpy as np
# from datasets.MN40 import MN40
from utils.config import *
import argparse
import sys
import os.path as osp
import time
import os
from utils.tensorboard_logger import TensorboardLogger
from utils.metric_logger import *
from model.build_model import build_model
from dataset.build_dataset import build_data_loader
from utils.solver import build_optimizer, build_scheduler
from utils.torch_utils import set_random_seed
from utils.checkpoint import Checkpointer
from utils import config
from utils.logger import get_logger


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


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                curr_epoch,
                tensorboard_logger,
                log_period=1,
                output_dir="",
                ):
    # logger = logging.getLogger("ModelZoo.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()

    for iteration, (input, target) in enumerate(data_loader):
        data_time = time.time() - end
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch if isinstance(v, torch.Tensor)}
        input = input.cuda()
        target = target.cuda()

        preds = model(input, )
        optimizer.zero_grad()

        loss_dict = loss_fn(preds, target)
        metric_dict = metric_fn(preds, input, target)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)

        losses.backward()

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "EPOCH: {epoch:2d}",
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    epoch=curr_epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
            tensorboard_logger.add_scalars(loss_dict, curr_epoch * total_iteration + iteration, prefix="train.loss")
            tensorboard_logger.add_scalars(metric_dict, curr_epoch * total_iteration + iteration, prefix="train.metric")

    return meters


def train(cfg, output_dir=""):
    # logger = logging.getLogger("ModelZoo.trainer")

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)

    checkpoint_data = checkpointer.load(cfg.GLOBAL.TRAIN.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.GLOBAL.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_data_loader(cfg, cfg.GLOBAL.DATASET, mode="train")
    val_period = cfg.GLOBAL.VAL.VAL_PERIOD
    # val_data_loader = build_data_loader(cfg, mode="val") if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.GLOBAL.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    # best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    # best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()
        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   data_loader=train_data_loader,
                                   optimizer=optimizer,
                                   curr_epoch=epoch,
                                   tensorboard_logger=tensorboard_logger,
                                   log_period=cfg.GLOBAL.TRAIN.LOG_PERIOD,
                                   output_dir=output_dir,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            # checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        '''
        # validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
            val_meters = validate_model(model,
                                        loss_fn,
                                        metric_fn,
                                        image_scales=cfg.MODEL.VAL.IMG_SCALES,
                                        inter_scales=cfg.MODEL.VAL.INTER_SCALES,
                                        isFlow=(cur_epoch > cfg.SCHEDULER.INIT_EPOCH),
                                        data_loader=val_data_loader,
                                        curr_epoch=epoch,
                                        tensorboard_logger=tensorboard_logger,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        output_dir=output_dir,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

            # best validation
            cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
            if best_metric is None or cur_metric > best_metric:
                best_metric = cur_metric
                checkpoint_data["epoch"] = cur_epoch
                checkpoint_data[best_metric_name] = best_metric
                checkpointer.save("model_best", **checkpoint_data)
        '''

    logger.info("Train Finish！")
    # logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model


def main():
    global logger, cfg
    cfg = get_parser()

    output_dir = cfg.GLOBAL.OUTPUT_DIR
    if output_dir:
        msg = 'init_train'
        output_dir = osp.join(output_dir, "train_{}_{}".format(time.strftime("%m_%d_%H_%M_%S"), msg))
        os.mkdir(output_dir)

    logger = get_logger("Model-Zoo", output_dir, prefix="train")
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == '__main__':
    main()