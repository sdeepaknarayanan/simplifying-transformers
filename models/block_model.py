import argparse

import numpy as np
import logging
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
        self.out_hidden = config.block_out_hidden_features
        self.attn_heads = config.block_heads
        self.device = config.device
        self.d_k = config.block_d_k
        self.dropout = config.block_dropout

        self.conf = config
        self.epoch = 0

        self.attentionblock = BlockMultiHeadedAttention(
            self.attn_heads, self.hidden, d_k=self.d_k, dropout=self.dropout, out_linear_overwrite=self.out_hidden
        )
        # self.output_linear = nn.Linear(self.hidden, self.hidden)

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

    def load_state(self, load_optimizer: bool = True):
        if self.conf.block_load_checkpoint:
            path = (
                    self.conf.storage_directory + '/models/_checkpoints/' + self.conf.dataset + '/block_'
                    + str(self.conf.block) + '_' + str(self.conf.block_d_k) + '_' + str(self.conf.block_heads) + '/'
                    + 'BLOCK-latest.pth'
            )
            tmp = path

            print(f"Block Path: {path}")
            if os.path.exists(path):
                try:
                    from collections import OrderedDict

                    checkpoint = torch.load(path)
                    print("Loaded block checkpoint")
                    try:
                        state_dict = checkpoint['model_state_dict']
                        if self.conf.train and load_optimizer:
                            opt_state_dict = checkpoint['optimizer_state_dict']

                            new_opt_state_dict = OrderedDict()
                            for k, v in state_dict.items():

                                if k[0:7] != 'module.':
                                    new_opt_state_dict = opt_state_dict
                                    break
                                else:
                                    name = k[7:]  # remove `module.`
                                    new_opt_state_dict[name] = v

                            self.optimizer.load_state_dict(new_opt_state_dict)

                    except KeyError:
                        state_dict = checkpoint
                        logging.warning('Could not access ["model_state_dict"] for {t}, this is expected for foreign models'
                                        .format(t=tmp))

                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():

                        if k[0:7] != 'module.':
                            new_state_dict = state_dict
                            break
                        else:
                            name = k[7:]  # remove `module.`
                            new_state_dict[name] = v

                    try:
                        self.load_state_dict(new_state_dict)
                        print('Successfully loaded state dict for block model')
                    except Exception as e:
                        logging.warning('Failed to load state dict into model\n{e}'
                                        .format(e=e))

                    try:
                        self.epoch = checkpoint['epoch']
                    except KeyError as e:
                        logging.warning('Failed to load epoch from state dict, epoch is set to 0:\n{e}'
                                        .format(e=e))
                        self.epoch = 0

                except RuntimeError as e:
                    logging.warning('Failed to load state dict into model. No State was loaded and model is initialized'
                                    'randomly. Epoch is set to 0:\n{e}'
                                    .format(e=e))
                    self.epoch = 0

            else:
                if self.conf.block_model_checkpoint != '':
                    logging.warning('Could not find a state dict for block model at the location specified.')
                self.epoch = 0

    def save_model(self, running: bool = True):
        """
        Save the model state_dict to models/checkpoint/
        :param running: if true, the state dict is stored under 'latest.pth', overwritten each epoch during training,
            if false, the state_dict ist store with an epoch number and will not be overwritten during this training.
        :return:
        """
        file_dir = self.conf.storage_directory + '/models/_checkpoints/' + self.conf.dataset + '/'

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
        file_dir = self.conf.storage_directory + '/models/_checkpoints/' + self.conf.dataset + '/block_'\
                   + str(self.conf.block) + '_' + str(self.conf.block_d_k) + '_' + str(self.conf.block_heads) + '/'

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_name = self.conf.model + '-' + 'latest.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }, file_dir + file_name)

        if self.epoch % self.conf.save_checkpoint_every == 0:
            file_name = self.conf.model + '-' + str(self.epoch) + '.pth'
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
            }, file_dir + file_name)

        self.epoch += 1

    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--block_hidden_features', type=int, default=768, help='# of hidden features')
        parser.add_argument('--block_out_hidden_features', type=int, default=768, help='# of hidden features')
        parser.add_argument('--block_heads', type=int, default=12, help='# of attention heads')
        parser.add_argument('--block_d_k', type=int, default=64, help='length of the key/query/value for each head')
        parser.add_argument('--block_dropout', type=float, default=0.1, help='dropout probability')
        parser.add_argument('--block_load_checkpoint', dest='block_load_checkpoint', action='store_true')
        parser.add_argument('--block_no_checkpoint', dest='block_load_checkpoint', action='store_false')
        parser.set_defaults(load_checkpoint=False)

        return parser
