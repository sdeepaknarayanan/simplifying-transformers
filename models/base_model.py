import torch
import logging
import os
from abc import abstractmethod
from typing import overload, Tuple


class BaseModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @overload
    def forward(self, data):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @staticmethod
    def extend_parser(parser):
        return parser


class BaseModel(BaseModule):
    @overload
    def forward(self, x, segment_info):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.sample = None
        self.epoch = 0

    def preprocess_data(self, data):
        return data

    def train_batch(self, data, criterion):
        """
        Predict for the current batch, compute loss and optimize model weights
        :param data: dictionary containing entries image, label & mask
        :param criterion: custom loss which computes a loss for data objects
        :return: current loss as float value
        """
        self.train()

        # send data-points to device (GPU)
        for key, value in data.items():
            data.update({key: value.to(self.conf.device)})

        # preprocess data batch with kornia
        data = self.preprocess_data(data)

        # make prediction for the current batch
        # data.update({'pred': self.forward(data['bert_input'], data['segment_label'])})
        prediction = self.forward(data['bert_input'], data['segment_label'])
        # compute loss, followed by backward pass and optimization step to improve weights
        loss = criterion(prediction.transpose(1, 2), data['bert_label'])

        for param in self.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def initialize_sample(self, batch):
        # store a test batch which can be used to print samples during training. this sample is used in 'print_sample()'
        self.sample = batch

    @torch.no_grad()
    def evaluate(self, data, criterion=None) -> Tuple[dict, float]:
        self.eval()

        # send data-points to device (GPU)
        for key, value in data.items():
            data.update({key: value.to(self.conf.device)})

        # make prediction for the current batch
        data.update({'pred': self.forward(data['bert_input'], data['segment_label'])})
        # compute loss if one is provided. make sure the losses output their values to some log, as no loss value is
        # returned here
        loss = criterion(data['pred'].transpose(1, 2), data['bert_label']).item() if criterion is not None else None

        return data, loss

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

    def print_sample(self):
        """
        Predict the sample batch with the current model weights and store the result in the out folder
        :return:
        """

        # TODO: implement
        # load and predict the sample batch
        data = self.sample.copy()
        data, loss = self.evaluate(data)

    def load_state(self):
        path = self.conf.model_checkpoint
        tmp = path

        if os.path.exists(path):
            try:
                from collections import OrderedDict

                checkpoint = torch.load(path)
                try:
                    state_dict = checkpoint['model_state_dict']
                    if self.conf.train:
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
                    print('Successfully loaded state dict')
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
            if self.conf.model_checkpoint != '':
                logging.warning('Could not find a state dict at the location specified.')
            self.epoch = 0

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
    def extend_parser(parser):
        parser.add_argument('--model_checkpoint', type=str, default='', help=
                            'path to a model_state_dict which will be loaded into the model before training/eval')

        return parser
