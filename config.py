import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 510510
        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False
        model_name = "sonnet"

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True
        act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [76, 76] # patch shape at output of network

        # self.dataset_name = "monusac"
        self.dataset_name = "glysac" # extracts dataset info from dataset.py
        # self.dataset_name = "consep"

        self.log_dir = "logs/" # where checkpoints will be saved
        nt_class_num = None
        if self.type_classification:
            nt_class_num = 4 if self.dataset_name == "glysac" else 5  # for consep and monusac # number of nuclear types (including background)
        num_classes = 1024  # number of nuclear types (including background)
        nf_class_num = 2
        no_class_num = 16
        # paths to training and validation patches
        self.valid_dir_list = [
            f"./dataset/training_data/{self.dataset_name}/valid/540x540_164x164"
        ]
        self.train_dir_list = [
            f"./dataset/training_data/{self.dataset_name}/train/540x540_164x164"
        ]
        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(num_classes, nf_class_num, no_class_num, nt_class_num)
