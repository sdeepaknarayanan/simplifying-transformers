import logging

import torch

import datasets
import models
from config.train_config import TrainConfig


def main(conf):
    vocab = datasets.get_vocab(conf)

    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)

    train_loader = train_dataset.get_data_loader()

    merged = models.get(model_name="MergedRetrainedBert")(config, vocab_size=len(vocab))
    merged.load_state(load_optimizer=False, overwrite_path='models/_checkpoints/wikitext2/MergedRetrainedBert-latest.pth')
    merged.eval()

    base_model = models.get(model_name="BERTLM")(config=conf, vocab_size=len(vocab))
    base_model.load_state(load_optimizer=False, overwrite_path="models/_checkpoints/wikitext2/BERTLM-latest.pth")
    base_model = base_model.to(conf.device)
    base_model.eval()

    logging.log(logging.INFO, "Initialized")

    epoch = merged.epoch
    criterion = torch.nn.CrossEntropyLoss()

    while epoch < conf.epochs + 1:

        for index, data in enumerate(train_loader):

            x = data['bert_input'].to(conf.device)
            segment_info = data['segment_label'].to(conf.device)
            y = merged(x, segment_info)

            y = y[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]

            teacher_pred = base_model(x, segment_info)
            teacher_pred = teacher_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]

            merged.zero_grad()
            merged.optimizer.zero_grad()

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                teacher_pred = torch.softmax(teacher_pred, dim=-1)

            loss = criterion(y, teacher_pred)
            loss.backward()
            merged.optimizer.step()

            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval Current loss: {l:.5f}'
                  .format(e=epoch, b=index + 1, t=len(train_loader), l=loss))


if __name__ == "__main__":

    option = TrainConfig()
    option.print()
    config = option.get()

    main(config)
