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
    with tqdm.tqdm(test_loader, unit="batch") as tq_loader:
        for index, data in enumerate(tq_loader):
            tq_loader.set_description(f"Batch: {index}")
            data, (f1, ce, p) = model.evaluate(data)
            f1_accumulated = f1_accumulated + f1
            ce_accumulated = ce_accumulated + ce
            p_accumulated = p_accumulated + p
            masked_prediction = data['pred'][torch.arange(data['pred'].size(0)), data['mask_index']]
            predicted_label = torch.argmax(masked_prediction, dim=1)
            gt_label = data['bert_label'][torch.arange(data['pred'].size(0)), data['mask_index']]
            tq_loader.set_postfix(f1=f1_accumulated / (index + 1),
                                  ce=ce_accumulated / (index + 1),
                                  perplexity=p_accumulated / (index + 1))


if __name__ == "__main__":

    # gather necessary config data from different parsers and combine.
    option = TestConfig()
    option.print()
    config = option.get()

    # start training
    main(config)