import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from .net_desc import create_model
from .run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output


# TODO: training config only ?
# TODO: switch all to function name String for all option
def get_config(num_classes, nf_class_num, no_class_num, nt_class_num=None):
    return {
        # ------------------------------------------------------------------
        # ! All phases have the same number of run engine
        # phases are run sequentially from index 0 to N
        "phase_list": [
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            num_classes=num_classes,
                            nf_class_num=nf_class_num,
                            no_class_num=no_class_num,
                            nt_class_num=nt_class_num,
                            freeze=True
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        # "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingWarmRestarts(x, T_0=25, T_mult=1,
                        #                                                                          eta_min=1e-5,
                        #                                                                          last_epoch=-1),
                        "extra_info": {
                            "loss": {
                                "nf": {"fore_focal": 1, "binary_dice": 1},
                                "no": {"ordinal_focal": 1},
                                "nt": {"type_focal": 1},
                            },
                            "self_sp": False,
                            "weights": {
                                "nt": {
                                    "consep": [0.299, 2.348, 0.945, 0.638, 0.770],
                                    "glysac": [0.456, 1.408, 1.149, 0.987],
                                    "monusac":[0.315, 0.631, 0.875, 1.219, 1.959]
                                },
                                "nf": {
                                    "consep": [0.734, 1.266],
                                    "glysac": [0.735, 1.265],
                                    "monusac": [0.732, 1.268],
                                },
                                "no": {
                                    "consep": [],
                                    "glysac": [],
                                    "monusac": []
                                }
                            }
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        # "pretrained": "./models/sonnet/efficientnet_pytorch/efficientnetb0.pth",
                        'pretrained': None,
                    },
                },
                "batch_size": {"train": 16, "valid": 16,},  # engine name : value
                # "batch_size": {"train": 8, "valid": 16, },  # engine name : value
                "nr_epochs": 50,
            },
            {
                "run_info": {
                    # may need more dynamic for each network
                    "net": {
                        "desc": lambda: create_model(
                            num_classes=num_classes,
                            nf_class_num=nf_class_num,
                            no_class_num=no_class_num,
                            nt_class_num=nt_class_num,
                            freeze=False
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {  # should match keyword for parameters within the optimizer
                                "lr": 1.0e-4,  # initial learning rate,
                                "betas": (0.9, 0.999),
                            },
                        ],
                        # learning rate scheduler
                        # "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(x, 25),
                        "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingWarmRestarts(x, T_0=25, T_mult=1,
                                                                                                 eta_min=1e-5,
                                                                                                 last_epoch=-1),
                        "extra_info": {
                            "loss": {
                                "nf": {"fore_focal": 1, "binary_dice": 1},
                                "no": {"ordinal_focal": 1},
                                "nt": {"type_focal": 1},
                            },
                            "self_sp": True,
                            "weights": {
                                "nt": {
                                    "consep": [0.299, 2.348, 0.945, 0.638, 0.770],
                                    "glysac": [0.456, 1.408, 1.149, 0.987],
                                    "monusac":[0.315, 0.631, 0.875, 1.219, 1.959]
                                },
                                "nf": {
                                    "consep": [0.734, 1.266],
                                    "glysac": [0.735, 1.265],
                                    "monusac": [0.732, 1.268],
                                },
                                "no": {
                                    "consep": [],
                                    "glysac": [],
                                    "monusac":[]
                                }
                            }
                        },
                        # path to load, -1 to auto load checkpoint from previous phase,
                        # None to start from scratch
                        "pretrained": -1,
                    },
                },
                "batch_size": {"train": 4, "valid": 8}, # batch size per gpu
                "nr_epochs": 50,
            },
        ],
        # ------------------------------------------------------------------
        # TODO: dynamically for dataset plugin selection and processing also?
        # all enclosed engine shares the same neural networks
        # as the on at the outer calling it
        "run_engine": {
            "train": {
                # TODO: align here, file path or what? what about CV?
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 16,  # number of threads for dataloader
                "run_step": train_step,  # TODO: function name or function variable ?
                "reset_per_run": False,
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        # LoggingGradient(), # TODO: very slow, may be due to back forth of tensor/numpy ?
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",  # whats about compound dataset ?
                "nr_procs": 16,  # number of threads for dataloader
                "run_step": valid_step,
                "reset_per_run": True,  # * to stop aggregating output etc. from last run
                # callbacks are run according to the list order of the event
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        # TODO: is there way to preload these ?
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nt_class_num)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }
