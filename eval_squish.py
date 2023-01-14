import logging

import torch
from operator import add
from transformers import AutoTokenizer, AutoModelForMaskedLM

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

    squished = models.get(config.model)(config, vocab_size=len(vocab))
    squished.load_state(load_optimizer=False)
    squished.eval()

    base_model = models.get(model_name="BERTLM")(config=conf, vocab_size=len(vocab))
    base_model.load_state(load_optimizer=False, overwrite_path="models/_checkpoints/wikitext2/BERTLM-latest.pth")
    base_model = base_model.to(conf.device)
    base_model.eval()

    tokenizer_large = AutoTokenizer.from_pretrained("bert-large-uncased")
    model_large = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
    model_large = model_large.to(conf.device)
    model_large.eval()

    logging.log(logging.INFO, "Initialized")

    retrained_score = [0, 0, 0]
    teacher_score = [0, 0, 0]
    total = 0
    breaking_point = int(.2 * len(test_loader))

    with open('out_squish/squish.txt', 'w', encoding='utf-8') as file:
        file.write('{r:>20} | {t:>20} || {w}\n'.format(r="Retrained", t="Teacher", w="Top 10 from Bert Large"))

        # print('{r:>20} | {t:>20} || {w}'.format(r="Retrained", t="Teacher", w="Top 10 from Bert Large"))
        for index, data in enumerate(test_loader):

            if index > breaking_point:
                break

            retrained_pred = squished(data['bert_input'].to(conf.device), data['segment_label'].to(config.device))
            retrained_pred = retrained_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            retrained_pred = torch.argmax(retrained_pred, dim=1)

            teacher_pred = base_model(data['bert_input'].to(conf.device), data['segment_label'].to(config.device))
            teacher_pred = teacher_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            teacher_pred = torch.argmax(teacher_pred, dim=1)

            for sample_index in range(data['bert_input'].size(0)):
                sentence = vocab.from_seq(data['bert_input'][sample_index], join=True)
                large_tokens = tokenizer_large(sentence, return_tensors="pt").to(conf.device)
                large_pred = model_large(**large_tokens).logits
                large_pred = large_pred[0][data['mask_index'][sample_index]]
                _, top_indices = large_pred.topk(10)
                top_words = tokenizer_large.decode(top_indices)

                retrained_word = vocab.itos[retrained_pred[sample_index]]
                teacher_word = vocab.itos[teacher_pred[sample_index]]
                file.write('{r:>20} | {t:>20} || {w}\n'.format(r=retrained_word, t=teacher_word, w=top_words))
                # print('{r:>20} | {t:>20} || {w}'.format(r=retrained_word, t=teacher_word, w=top_words))
                top_1 = retrained_word in top_words[0:1]
                top_5 = retrained_word in top_words[0:5]
                top_10 = retrained_word in top_words[0:10]
                retrained_score = list(map(add, retrained_score, [top_1, top_5, top_10]))

                top_1 = teacher_word in top_words[0:1]
                top_5 = teacher_word in top_words[0:5]
                top_10 = teacher_word in top_words[0:10]
                teacher_score = list(map(add, teacher_score, [top_1, top_5, top_10]))

                total += 1
        file.write('Teacher Scores    Top 1: {t1}, Top 5: {t2}, Top 10: {t3}\n'.format(
            t1=teacher_score[0], t2=teacher_score[1], t3=teacher_score[2]))
        # print('Teacher Scores    Top 1: {t1}, Top 5: {t2}, Top 10: {t3}'.format(
        #     t1=teacher_score[0], t2=teacher_score[1], t3=teacher_score[2]))
        file.write('Retrained Scores: Top 1: {r1}, Top 5: {r2}, Top 10: {r3}\n'.format(
            r1=retrained_score[0], r2=retrained_score[2], r3=retrained_score[2]))
        # print('Retrained Scores: Top 1: {r1}, Top 5: {r2}, Top 10: {r3}'.format(
        #     r1=retrained_score[0], r2=retrained_score[2], r3=retrained_score[2]))
        file.close()
        exit(0)


if __name__ == "__main__":
    # gather necessary config data from different parsers and combine.
    option = TestConfig()
    option.print()
    config = option.get()

    # start training
    main(config)
