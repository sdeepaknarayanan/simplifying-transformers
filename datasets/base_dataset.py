import argparse
from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        return parser
