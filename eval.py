import logging

import torch

import datasets
import models
from config.test_config import TestConfig



def main(conf):

    vocab = datasets.get_vocab(conf)

    # load the dataset specified with --dataset_name & get data loaders
    # train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

    # train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()

    model = models.get(model_name=conf.model)(config=conf, vocab_size=len(vocab))
    model.initialize_sample(batch=next(iter(test_loader)))

    model.load_state()

    logging.log(logging.INFO, "Initialized")

    for index, data in enumerate(test_loader):
        data, _ = model.evaluate(data)

        for ind in range(data['pred'].size(0)):

            data['pred'][ind, data["mask_index"][ind], 101] = -1e10 # disregard CLS token
            data['pred'][ind, data["mask_index"][ind], 103] = -1e10
            # data['pred'][ind, data["mask_index"][ind], 105] = -1e10

            sentence = vocab.from_seq(torch.masked_select(data['bert_input'][ind], data['segment_label'][ind].bool()))
            predicted = vocab.from_index(torch.argmax(data['pred'][ind], dim=1)[data['mask_index'][ind]])
            gt = vocab.from_index(data['bert_label'][ind][data['mask_index'][ind]])
            print("Input: ", sentence)
            print(f"Prediction: {predicted}, GT: {gt}")


        exit()


if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = TestConfig()
    option.print()
    config = option.get()

    # start training
    main(config)