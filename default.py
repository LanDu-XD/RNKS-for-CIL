from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from yacs.config import CfgNode as CN


_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "default"
_C.RESUME_MODEL = ""
_C.RESUME_MODE = "all"
_C.CPU_MODE = False

# ----- BACKBONE BUILDER -----
_C.BACKBONE = CN()
_C.BACKBONE.FREEZE = False
_C.BACKBONE.PRETRAINED_MODEL = ""

# ----- TRAIN BUILDER -----
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.MAX_EPOCH = 200
_C.TRAIN.SHUFFLE = True
_C.TRAIN.NUM_WORKERS = 8

# ----- SAMPLER BUILDER -----
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = "SGD"
_C.TRAIN.OPTIMIZER.BASE_LR = 0.5
_C.TRAIN.OPTIMIZER.PROXY_LR = 1.0
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4


def update_config(cfg):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)

    cfg.freeze()