import logging

import torch.backends.cudnn

import datasets
import models
torch.backends.cudnn.benchmark = True
from config.train_config import BlockTrainConfig


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
    block_model.train()

    base_model = models.get(model_name="BERTLM")(config=conf, vocab_size=len(vocab))
    base_model.load_state(load_optimizer=False, overwrite_path="models/_checkpoints/wikitext2/BERTLM-latest.pth")
    base_model.eval()

    logging.log(logging.INFO, "Initialized")

    # load loss and evaluation metric
    criterion = torch.nn.CrossEntropyLoss()

    epoch = block_model.epoch

    # main training loop
    while epoch < conf.epochs + 1:

        # Train on training-set
        for index, data in enumerate(train_loader):

            block_model.zero_grad()

            x = data['bert_input'].to(conf.device)

            segment_info = data['segment_label'].to(conf.device)

            teacher_output = base_model(x, segment_info)
            teacher_output = teacher_output[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            teacher_output = torch.nn.functional.softmax(teacher_output, dim=1)

            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            x = base_model.bert.embedding(x, segment_info)

            for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                if layer == conf.block:
                    break
                x, _ = transformer.forward(x, mask)

            y = transformer.input_sublayer(x, lambda _x: block_model.forward(_x, mask)[0])

            y = transformer.output_sublayer(y, transformer.feed_forward)
            y = transformer.dropout(y)

            for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                if layer <= conf.block:
                    continue
                y, _ = transformer.forward(y, mask)

            y = base_model.mask_lm(y)

            y = y[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            y = torch.nn.functional.softmax(y, dim=1)
            loss = criterion(y, teacher_output)
            loss.backward()
            block_model.optimizer.step()

            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}]. Current loss: {l:.5f}'.format(
                e=epoch, b=index + 1, t=len(train_loader), l=loss))

        del x, y, mask, segment_info, loss

        block_model.save_and_step()

        torch.cuda.empty_cache()
        # Evaluate on test-set.
        for index, data in enumerate(test_loader):

            x = None
            y = None
            with torch.no_grad():
                x = data['bert_input'].to(conf.device)
                segment_info = data['segment_label'].to(conf.device)
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
                x = base_model.bert.embedding(x, segment_info)

                for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                    if layer == conf.block:
                        break
                    x = transformer.forward(x, mask)

                y = transformer.attention.forward(x, x, x, mask=mask)

            x, loss = block_model.evaluate((x, y), criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval Current loss: {l:.5f}'.format(e=epoch, b=index + 1,
                                                                                          t=len(test_loader), l=loss))

        del x, y, mask, segment_info, loss
        torch.cuda.empty_cache()
        # Write the accumulated losses to logs and reset and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='test')

        # Save sample images containing prediction and label side by side
        if conf.print_samples:

            data = next(iter(test_loader))

            x = data['bert_input'].to(conf.device)
            segment_info = data['segment_label'].to(conf.device)
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            x = base_model.bert.embedding(x, segment_info)

            for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                if layer == conf.block:
                    break
                x = transformer.forward(x, mask)

            y = transformer.input_sublayer(x, lambda _x: block_model.forward(_x))
            y = transformer.output_sublayer(y, transformer.feed_forward)
            y = transformer.dropout(y)

            for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                if layer <= conf.block:
                    continue
                y = transformer.forward(y, mask)

            y = base_model.mask_lm(y)

            teacher_pred = base_model(data['bert_input'].to(conf.device), data['segment_label'].to(config.device))

            masked_prediction = y[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            predicted_label = torch.argmax(masked_prediction, dim=1)

            masked_teacher_pred = teacher_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            masked_teacher_pred = torch.argmax(masked_teacher_pred, dim=1)

            gt_label = data['bert_label'][torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            for i in range(data['bert_input'].size(0)):
                print('GT: {g:>15}, TEACHER: {t:>15}, PRED: {p:>15}'.format(
                    g=vocab.itos[gt_label[i]],
                    t=vocab.itos[masked_teacher_pred[i]],
                    p=vocab.itos[predicted_label[i]]),
                    vocab.from_seq(data['bert_input'][i], join=True)
                )
            del gt_label, masked_teacher_pred, predicted_label, masked_prediction, teacher_pred, y, x, segment_info, mask

        torch.cuda.empty_cache()
        epoch += 1


if __name__ == "__main__":
    # gather necessary config data from different parsers and combine.
    option = BlockTrainConfig()
    option.print()
    config = option.get()

    # start training
    train_block(config)
