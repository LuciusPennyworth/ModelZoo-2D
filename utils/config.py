from yacs.config import CfgNode as CN
from yacs.config import load_cfg


_C = CN()

# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = 1

# -----------------------------------------------------------------------------
# GLOBAL
# -----------------------------------------------------------------------------

_C.GLOBAL = CN()

_C.GLOBAL.TRAIN = CN()
_C.GLOBAL.TRAIN.ROOT_DIR = ""
_C.GLOBAL.TRAIN.WEIGHT = ""
_C.GLOBAL.TRAIN.BATCH_SIZE = 8
_C.GLOBAL.TRAIN.CHECKPOINT_PERIOD = 48
_C.GLOBAL.TRAIN.LOG_PERIOD = 50

_C.GLOBAL.VAL = CN()
_C.GLOBAL.VAL.ROOT_DIR = ""
_C.GLOBAL.VAL.VAL_PERIOD = 0

_C.GLOBAL.TEST = CN()
_C.GLOBAL.TEST.ROOT_DIR = ""
_C.GLOBAL.TEST.WEIGHT = ""
_C.GLOBAL.TEST.BATCH_SIZE = 1

_C.GLOBAL.OUTPUT_DIR = ""
_C.GLOBAL.INPUT_SIZE = 784
_C.GLOBAL.NUM_WORKERS = 4
_C.GLOBAL.MAX_EPOCH = 16
_C.GLOBAL.CLASSES = 10
_C.GLOBAL.TRAIN_GPU = [0,1,2,3]
_C.GLOBAL.MODEL = "net"
_C.GLOBAL.DATASET = "CIFAR10"

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

# _C.MODEL = CN()
# _C.MODEL.WEIGHT = ""
#
# _C.MODEL.EDGE_CHANNELS = ()
# _C.MODEL.FLOW_CHANNELS = (64, 64, 16, 1)
# _C.MODEL.NUM_VIRTUAL_PLANE = 48
# _C.MODEL.IMG_BASE_CHANNELS = 8
# _C.MODEL.VOL_BASE_CHANNELS = 8
#
# _C.MODEL.VALID_THRESHOLD = 8.0
#
# _C.MODEL.TRAIN = CN()
# _C.MODEL.TRAIN.IMG_SCALES = (0.125, 0.25)
# _C.MODEL.TRAIN.INTER_SCALES = (0.75, 0.375)
#
# _C.MODEL.VAL = CN()
# _C.MODEL.VAL.IMG_SCALES = (0.125, 0.25)
# _C.MODEL.VAL.INTER_SCALES = (0.75, 0.375)
#
# _C.MODEL.TEST = CN()
# _C.MODEL.TEST.IMG_SCALES = (0.125, 0.25, 0.5)
# _C.MODEL.TEST.INTER_SCALES = (1.0, 0.75, 0.15)

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #

_C.OPTIM = CN()

# Type of optimizer
_C.OPTIM.TYPE = "RMSprop"
# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.OPTIM.BASE_LR = 0.001
_C.OPTIM.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.OPTIM.RMSprop = CN()
_C.OPTIM.RMSprop.alpha = 0.9

_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.momentum = 0.9

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ""

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1





def load_cfg_from_file(cfg_filename):
    """Load config from a file

    Args:
        cfg_filename (str):

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)

    cfg_template = _C
    cfg_template.merge_from_other_cfg(cfg)
    return cfg_template





