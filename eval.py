import logging

import torch
import tqdm

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
    f1_accumulated = 0.
    ce_accumulated = 0.
    p_accumulated = 0.
    i = 0
    with tqdm.tqdm(test_loader, unit="batch") as tq_loader:
        for index, data in enumerate(tq_loader):
            i = index
            tq_loader.set_description(f"Batch: {index}")
            data, (f1, ce, p) = model.evaluate(data)
            f1_accumulated = f1_accumulated + f1
            ce_accumulated = ce_accumulated + ce
            p_accumulated = p_accumulated + p
            tq_loader.set_postfix(f1=f1_accumulated / (index + 1),
                                  ce=ce_accumulated / (index + 1),
                                  perplexity=p_accumulated / (index + 1))

    print(f"Cross Entropy: {ce_accumulated}, F1: {f1_accumulated}, Perplexity: {p_accumulated}")


if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = TestConfig()
    option.print()
    config = option.get()

    # start training
    main(config)