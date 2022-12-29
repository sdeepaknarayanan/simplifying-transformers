import argparse
from typing import Literal

import tqdm
import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDataset

class BlockDataset(BaseDataset):
  """
  """
  def __init__(self, config, split: Literal["train", "test", "val"] = "train"):
    super(BlockDataset, self).__init__()

    self.config = config

    match split:
      case "train":
        self.data_path = config.train_dataset
      case "test":  
        self.data_path = config.test_dataset
      case "val":
        self.data_path = config.val_dataset    
        
    self.data =np.load(self.data_path).astype(np.single)
    

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, item):
    return self.data[item][0], self.data[item][1]

  def get_data_loader(self):
        return DataLoader(self, batch_size=self.config.batch_size, num_workers=self.config.num_workers)


  @staticmethod
  def extend_parser(parser) -> argparse.ArgumentParser:
    parser.add_argument('--train_dataset', type=str, default='data/blockdata/block_testdata.npy')
    parser.add_argument('--test_dataset', type=str, default='data/blockdata/block_testdata.npy')
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    return parser

