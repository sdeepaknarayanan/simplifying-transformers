import logging

import torch.backends.cudnn

import datasets
import models
from config.train_config import TrainConfig
# torch.backends.cudnn.benchmark = True

from train_block import train_block
from config.train_config import BlockTrainConfig

def train_block(conf):
    """ 
    Training for a single attention block. Similary structured as main in train.py but just for one block.

    :param conf: config object from arg parser containing all the block training configuration
    :return:
    
    """

    # load the dataset specified with --dataset_name & get data loaders
    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, split="test")

    train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()

    # load the model specified with --model_name
    model = models.get(model_name=conf.model)(config=conf)

    model.initialize_sample(batch=next(iter(test_loader)))

    logging.log(logging.INFO, "Initialized")

    # load loss and evaluation metric 
    criterion = torch.nn.MSELoss()

    # if a model checkpoint was loaded then the epoch is set to the epoch the model was saved on (continue training)
    epoch = model.epoch

    # main training loop
    while epoch < conf.epochs + 1:

        # Train on training-set
        for index, data in enumerate(train_loader):
            loss = model.train_batch(data, criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}]. Current loss: {l:.5f}'.format(
                e=epoch, b=index+1, t=len(train_loader), l=loss))

        # Save model checkpoint & write the accumulated losses to logs and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='train')
        model.save_and_step()

        # Evaluate on test-set.
        for index, data in enumerate(test_loader):
            model.evaluate(data, criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval'.format(e=epoch, b=index+1, t=len(test_loader)))

        # Write the accumulated losses to logs and reset and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='test')

        # Save sample images containing prediction and label side by side
        if conf.print_samples:
            model.print_sample()
        epoch += 1



if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = BlockTrainConfig()
    option.print()
    config = option.get()

    # start training
    train_block(config)