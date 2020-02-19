import datetime
import pathlib
import torch
import torchvision

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg


def main():
    # set up logger with timestamp from start of script
    start_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_root = pathlib.Path('..', '..', 'logs', 'train_particles')
    log_dir = pathlib.Path(log_root, 'train_particles_{}.log'.format(start_timestamp))
    setup_logger(output=log_dir)


