class PrinterLogger:
    def __init__(self, logger):
        self.logger = logger

    def print_and_log(self, text):
        self.logger.info(text)
        print(text)

    def info(self, text):
        self.logger.info(text)


def seed_everything(seed):
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    import random

    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from torch_geometric import seed_everything

    seed_everything(seed)
