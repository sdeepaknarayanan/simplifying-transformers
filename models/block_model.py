import argparse

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import os

from models.base_model import BaseModule
from models.attention.block import BlockMultiHeadedAttention
from models.bert import ScheduledOptim

from typing import overload, Tuple


class BLOCK(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.hidden = config.block_hidden_features
        self.attn_heads = config.block_heads
        self.device = config.device
        self.d_k = config.block_d_k
        self.dropout = config.block_dropout

        self.conf = config
        self.epoch = 0

        self.attentionblock = BlockMultiHeadedAttention(self.attn_heads, self.hidden, d_k =self.d_k, dropout = self.dropout)
        self.output_linear = nn.Linear(self.hidden, self.hidden)

        self.optimizer = Adam(
            self.parameters(),
            lr=self.conf.lr,
            betas=(self.conf.adam_beta1, self.conf.adam_beta2),
            weight_decay=self.conf.adam_weight_decay
        )

        self.optim_schedule = ScheduledOptim(
            self.optimizer,
            self.hidden,
            n_warmup_steps=self.conf.warmup_steps
        )

        self.to(config.device)

    def forward(self, x):
        return self.attentionblock(x,x,x)

    def train_batch(self, data, criterion):
        """
        Predict for the current batch, compute loss and optimize model weights
        :param data: dictionary containing entries image, label & mask
        :param criterion: custom loss which computes a loss for data objects
        :return: current loss as float value
        """
        self.train()

        x,y = data

        # send data-points to device (GPU)
        x = x.to(self.device)
        y = y.to(self.device)

        for param in self.parameters():
            param.grad = None

        # make prediction for the current batch
        prediction = self.forward(x)

        loss = criterion(prediction, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def initialize_sample(self, batch):
        # store a test batch which can be used to print samples during training. this sample is used in 'print_sample()'
        self.sample = batch
    
    @torch.no_grad()
    def evaluate(self, data, criterion=None) -> Tuple[float, float]:

        self.eval()


        x,y = data


         # send data-points to device (GPU)
        x = x.to(self.device)
        y = y.to(self.device)

        # make prediction for the current batch
        x = self.forward(x)
        # compute loss if one is provided. make sure the losses output their values to some log, as no loss value is
        # returned here
        loss = criterion(x, y)

        return x, loss
    
    def save_model(self, running: bool = True):
        """
        Save the model state_dict to models/checkpoint/
        :param running: if true, the state dict is stored under 'latest.pth', overwritten each epoch during training,
            if false, the state_dict ist store with an epoch number and will not be overwritten during this training.
        :return:
        """
        file_dir = self.conf.storage_directory + '/models/_checkpoints/' + self.conf.dataset + '_' +\
                   self.conf.dataset + '/'

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if running:
            file_name = self.conf.model + '-' + 'latest.pth'
        else:
            file_name = self.conf.model + '-' + str(self.epoch) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }, file_dir + file_name)
    
    def save_and_step(self):
        """
        Save checkpoints and decrease learn rate, should be called at the end of every training epoch
        """
        # save the most recent model to enable continuation if necessary
        self.save_model()

        # save checkpoint
        if self.epoch % self.conf.save_checkpoint_every == 0:
            self.save_model(running=False)

        self.epoch += 1

    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--block_hidden_features', type=int, default=768, help='# of hidden features')
        parser.add_argument('--block_heads', type=int, default=12, help='# of attention heads')
        parser.add_argument('--block_d_k', type=int, default=64, help='length of the key/query/value for each head')
        parser.add_argument('--block_dropout', type=float, default=0.1, help='dropout probability')
        parser.add_argument('--block_model_checkpoint', type=str, default='', help=
                            'path to a model_state_dict which will be loaded into the model before training/eval')

        return parser
