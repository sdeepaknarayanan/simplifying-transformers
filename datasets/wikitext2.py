import argparse
from typing import Literal

import tqdm
import torch
import random
import numpy as np

from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDataset

from transformers import BertTokenizerFast, BertTokenizer


class WikiText2Dataset(BaseDataset):
    def __init__(self, config, vocab, split: Literal["train", "test", "val"] = "train"):
        super(WikiText2Dataset, self).__init__()

        self.config = config
        self.vocab = vocab
        self.seq_len = config.seq_len

        self.on_memory = config.on_memory
        self.corpus_lines = config.corpus_lines

        if split == 'train':
            self.corpus_path = config.train_dataset
        if split == 'test':
            self.corpus_path = config.test_dataset
        if split == 'val':
            self.corpus_path = config.val_dataset

        self.encoding = config.encoding

        # self.tokenizer = BertTokenizerFast(vocab_file=config.vocab_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(self.corpus_path, "r", encoding=self.encoding) as f:
            if self.corpus_lines is None and not self.on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=self.corpus_lines):
                    self.corpus_lines += 1

            if self.on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=self.corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not self.on_memory:
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.random_sent(item)

        t1_random, t1_label, mask_index = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag

        t1 = [] + t1_random + []
        # t1 = [self.vocab.sos_index] + t1_random[:self.seq_len - 2] + [self.vocab.eos_index]

        t1_label = [] + t1_label + []
        # t1_label = [self.vocab.pad_index] + t1_label[:self.seq_len - 2] + [self.vocab.pad_index]

        segment_label = [1 for _ in range(len(t1))][:self.seq_len]
        #
        bert_input = t1[:self.seq_len]
        bert_label = t1_label[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        # print(padding)

        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "mask_index": mask_index
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):

        sentence = sentence.replace('<unk>', '[UNK]')
        token = self.tokenizer(sentence)
        tokens = token.input_ids

        if len(tokens) > 40:
            tokens = tokens[0:40]
            tokens[39] = self.vocab.eos_index

        index = np.random.randint(1, len(tokens) - 1)

        output_label = [0] * len(tokens)
        output_label[index] = tokens[index]
        tokens[index] = self.vocab.mask_index
        return tokens, output_label, index


    def random_sent(self, index):
        t1 = self.get_corpus_line(index)
        return t1

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0]

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

    def get_data_loader(self):
        return DataLoader(self,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers, shuffle= 2)

    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--seq_len', type=int, default=40, help="maximum sequence length")
        parser.add_argument('--train_dataset', type=str, default='data/wikitext2/train_data_single_sentence.txt')
        parser.add_argument('--test_dataset', type=str, default='data/wikitext2/test_data_single_sentence.txt')
        parser.add_argument('--val_dataset', type=str, default='data/wikitext2/valid_data_single_sentence.txt')
        parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
        parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
        parser.add_argument("--encoding", type=str, default='utf-8', help="text data encoding")
        parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

        return parser
