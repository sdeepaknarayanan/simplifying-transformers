from config.train_config import RetrainMergeConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM
from operator import add
from parent_bert import get_pretrained_berd
import logging

import torch.backends.cudnn

import datasets
import models

torch.backends.cudnn.benchmark = True
import time

def train_block(conf):
    """
    Training for a single attention block. Similary structured as main in train.py but just for one block.

    :param conf: config object from arg parser containing all the block training configuration
    :return:

    """
    with torch.no_grad():
        vocab = datasets.get_vocab(conf)
        # load the dataset specified with --dataset_name & get data loaders
        # train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
        test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

        # train_loader = train_dataset.get_data_loader()
        test_loader = test_dataset.get_data_loader()

        # load the blockmodel specified with --model_name
        block_model_0 = models.get(model_name=conf.model)(config=conf)
        block_model_1 = models.get(model_name=conf.model)(config=conf)
        block_model_2 = models.get(model_name=conf.model)(config=conf)
        block_model_3 = models.get(model_name=conf.model)(config=conf)
        block_model_4 = models.get(model_name=conf.model)(config=conf)
        block_model_5 = models.get(model_name=conf.model)(config=conf)
        
        best_model_0 = torch.load("merge_models_best/block_0")
        best_model_1 = torch.load("merge_models_best/block_2")
        best_model_2 = torch.load("merge_models_best/block_4")
        best_model_3 = torch.load("merge_models_best/block_6")
        best_model_4 = torch.load("merge_models_best/block_8")
        best_model_5 = torch.load("merge_models_best/block_10")
        
        block_model_0.load_state_dict(best_model_0['model_state_dict'])
        block_model_1.load_state_dict(best_model_1['model_state_dict'])
        block_model_2.load_state_dict(best_model_2['model_state_dict'])
        block_model_3.load_state_dict(best_model_3['model_state_dict'])
        block_model_4.load_state_dict(best_model_4['model_state_dict'])
        block_model_5.load_state_dict(best_model_5['model_state_dict'])

        base_model = get_pretrained_berd()

        block_model_0.eval()
        block_model_1.eval()
        block_model_2.eval()
        block_model_3.eval()
        block_model_4.eval()
        block_model_5.eval()
        base_model.eval()

        tokenizer_large = AutoTokenizer.from_pretrained("bert-large-uncased")
        model_large = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

        logging.log(logging.INFO, "Initialized")

        # load loss and evaluation metric

        # if a model checkpoint was loaded then the epoch is set to the epoch the model was saved on (continue training)

        retrained_score = [0, 0, 0]
        teacher_score = [0, 0, 0]
        total = 0
        print('{r:>20} | {t:>20} || {w}'.format(r="Retrained", t="Teacher", w="Top 10 from Bert Large"))
        num_batches = int(0.2*len(test_loader))
        print("Total Number of MBs:", num_batches)
        for index, data in enumerate(test_loader):
            start_time = time.time()
            if index > num_batches:
                break

            x = data['bert_input'].to(conf.device)
            segment_info = data['segment_label'].to(conf.device)
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            x = base_model.bert.embedding(x, segment_info)
            
            x = block_model_0.forward(x, mask)
            x = block_model_1.forward(x, mask)
            x = block_model_2.forward(x, mask)
            x = block_model_3.forward(x, mask)
            x = block_model_4.forward(x, mask)
            y = block_model_5.forward(x, mask)
            
            retrained_pred = base_model.mask_lm(y)
            retrained_pred = retrained_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            retrained_pred = torch.argmax(retrained_pred, dim=1)

            teacher_pred = base_model(data['bert_input'].to(conf.device), data['segment_label'].to(config.device))
            teacher_pred = teacher_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            teacher_pred = torch.argmax(teacher_pred, dim=1)

            for sample_index in range(data['bert_input'].size(0)):
                sentence = vocab.from_seq(data['bert_input'][sample_index], join=True)
                large_tokens = tokenizer_large(sentence, return_tensors="pt")
                large_pred = model_large(**large_tokens).logits
                large_pred = large_pred[0][data['mask_index'][sample_index]]
                _, top_indices = large_pred.topk(10)
                top_words = tokenizer_large.decode(top_indices)

                retrained_word = vocab.itos[retrained_pred[sample_index]]
                teacher_word = vocab.itos[teacher_pred[sample_index]]
                print('{r:>20} | {t:>20} || {w}'.format(r=retrained_word, t=teacher_word, w=top_words))
                top_1 = retrained_word in top_words[0:1]
                top_5 = retrained_word in top_words[0:5]
                top_10 = retrained_word in top_words[0:10]
                retrained_score = list(map(add, retrained_score, [top_1, top_5, top_10]))

                top_1 = teacher_word in top_words[0:1]
                top_5 = teacher_word in top_words[0:5]
                top_10 = teacher_word in top_words[0:10]
                teacher_score = list(map(add, teacher_score, [top_1, top_5, top_10]))

                total += 1
            end_time = time.time()
            print(f"Time: {end_time} - {start_time}")

        print('Teacher Scores    Top 1: {t1}, Top 5: {t2}, Top 10: {t3}'.format(
            t1=teacher_score[0], t2=teacher_score[1], t3=teacher_score[2]))
        print('Retrained Scores: Top 1: {r1}, Top 5: {r2}, Top 10: {r3}'.format(
            r1=retrained_score[0], r2=retrained_score[2], r3=retrained_score[2]))

        exit(0)


if __name__ == "__main__":
    # gather necessary config data from different parsers and combine.
    option = RetrainMergeConfig()
    option.print()
    config = option.get()

    # start training
    train_block(config)

