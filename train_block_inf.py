import logging

import torch.backends.cudnn

import datasets
import models
from config.train_config import TrainConfig
# torch.backends.cudnn.benchmark = True

from config.train_config import BlockTrainConfig

from parent_bert import get_pretrained_berd

def train_block(conf):
    """ 
    Training for a single attention block. Similary structured as main in train.py but just for one block.

    :param conf: config object from arg parser containing all the block training configuration
    :return:
    
    """

    vocab = datasets.get_vocab(conf)
    # load the dataset specified with --dataset_name & get data loaders
    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

    train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()


    # load the blockmodel specified with --model_name
    block_model = models.get(model_name=conf.model)(config=conf)

    block_model.initialize_sample(batch=next(iter(test_loader)))

    base_model = get_pretrained_berd()

    logging.log(logging.INFO, "Initialized")

    # load loss and evaluation metric 
    criterion = torch.nn.MSELoss()

    # if a model checkpoint was loaded then the epoch is set to the epoch the model was saved on (continue training)
    epoch = block_model.epoch

    # main training loop
    while epoch < conf.epochs + 1:

        # Train on training-set
        for index, data in enumerate(train_loader):

            x = None
            y = None

            with torch.no_grad():
                x = data['bert_input']
                segment_info = data['segment_label']
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
                x  = base_model.bert.embedding(x, segment_info)

                for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                    if(layer == config.block):
                        break
                    x = transformer.forward(x, mask)
            
                y = transformer.attention.forward(x,x,x, mask = mask) 

            loss = block_model.train_batch((x,y), criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}]. Current loss: {l:.5f}'.format(
                e=epoch, b=index+1, t=len(train_loader), l=loss))

        # Save model checkpoint & write the accumulated losses to logs and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='train')
        block_model.save_and_step()

        # Evaluate on test-set.
        for index, data in enumerate(test_loader):
            
            x = None
            y = None

            with torch.no_grad():
                x = data['bert_input']
                segment_info = data['segment_label']
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
                x  = base_model.bert.embedding(x, segment_info)

                for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                    if(layer == config.block):
                        break
                    x = transformer.forward(x, mask)
            
                y = transformer.attention.forward(x,x,x, mask = mask) 

            block_model.evaluate((x,y), criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval'.format(e=epoch, b=index+1, t=len(test_loader)))

        # Write the accumulated losses to logs and reset and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='test')

        # Save sample images containing prediction and label side by side
        if conf.print_samples:
            block_model.print_sample()
        epoch += 1



if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = BlockTrainConfig()
    option.print()
    config = option.get()

    # start training
    train_block(config)