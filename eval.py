import logging

import torch

import datasets
import models
from config.test_config import TestConfig



def main(conf):

    vocab = datasets.get_vocab(conf)
    # load the dataset specified with --dataset_name & get data loaders
    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

    train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()

    model = models.get(model_name=conf.model)(config=conf, vocab_size=len(vocab))
    model.initialize_sample(batch=next(iter(test_loader)))

    model.load_state()

    logging.log(logging.INFO, "Initialized")

    for index, data in enumerate(test_loader):

        data, _ = model.evaluate(data)
        for ind in range(config.batch_size):
            sentence = vocab.from_seq(data['bert_input'][ind])
            word = vocab.from_index(torch.argmax(data['pred'][ind]))
            print(sentence, word)
        exit()



if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = TestConfig()
    option.print()
    config = option.get()

    # start training
    main(config)