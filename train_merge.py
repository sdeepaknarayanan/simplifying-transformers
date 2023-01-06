import logging

import torch.backends.cudnn

import datasets
import models
from config.train_config import TrainConfig

torch.backends.cudnn.benchmark = True

from config.train_config import MergeTrainConfig

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

    #base_model = models.get(model_name="BERTLM")(config=conf, vocab_size=len(vocab))
    # for name, parameter in base_model.named_parameters():
    #     print(name, parameter.size())
    # exit()
    #base_model.load_state(load_optimizer=False)
    
    base_model = get_pretrained_berd()
    base_model.eval()

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
                x = data['bert_input'].to(config.device)
                segment_info = data['segment_label'].to(config.device)
                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
                x = base_model.bert.embedding(x, segment_info)

                for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                    if (layer == config.block):
                        temp_x = x
                    if(layer == (config.block + 2)):
                        break
                    x = transformer.forward(x, mask)


            loss = block_model.train_batch((temp_x, x), criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}]. Current loss: {l:.5f}'.format(
                e=epoch, b=index + 1, t=len(train_loader), l=loss))

        del x, temp_x, mask, segment_info, loss

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
                    if (layer == config.block):
                        temp_x = x
                    if(layer == (config.block + 2)):
                        break
                    x = transformer.forward(x, mask)

            x, loss = block_model.evaluate((temp_x, x), criterion)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval Current loss: {l:.5f}'.format(e=epoch, b=index + 1,
                                                                                          t=len(test_loader), l=loss))

        del x, temp_x, mask, segment_info, loss
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
                    if (layer == config.block):
                        break
                    x = transformer.forward(x, mask)

            y = block_model.forward(x)

            for layer, transformer in enumerate(base_model.bert.transformer_blocks):
                if layer <= conf.block+1:
                    pass
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
    option = MergeTrainConfig()
    option.print()
    config = option.get()

    # start training
    train_block(config)
