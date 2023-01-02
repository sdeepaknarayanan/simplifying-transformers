import logging

import torch

import datasets
import models
from config.train_config import TrainConfig
# torch.backends.cudnn.benchmark = True
from datasets.vocabulary import WordVocab


def main(conf):
    """
    Main training function. Coordinates training and calls all relevant functions.
    The functionality is mostly found on other files, so think of this as a coordination hub and not training logic.

    :param conf: config object from arg parser containing all the training configuration
    :return:
    """

    vocab = datasets.get_vocab(conf)
    # load the dataset specified with --dataset_name & get data loaders
    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

    train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()

    # load the model specified with --model_name
    model = models.get(model_name=conf.model)(config=conf, vocab_size=len(vocab))
    model.initialize_sample(batch=next(iter(test_loader)))
    print("Before Load")
    model.load_state()

    logging.log(logging.INFO, "Initialized")
    # load loss and evaluation metric
    criterion = torch.nn.NLLLoss(ignore_index=0)

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
    option = TrainConfig()
    option.print()
    config = option.get()

    # start training
    main(config)
