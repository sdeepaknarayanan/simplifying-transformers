import argparse
from typing import Literal

import tqdm
import torch
import random

from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDataset


class WikiText2Dataset(BaseDataset):
    def __init__(self, config, vocab, split: Literal["train", "test", "val"] = "train"):
        super(WikiText2Dataset, self).__init__()

        self.config = config
        self.vocab = vocab
        self.seq_len = config.seq_len

        self.on_memory = config.on_memory
        self.corpus_lines = config.corpus_lines
        match split:
            case "train":
                self.corpus_path = config.train_dataset
            case "test":
                self.corpus_path = config.test_dataset
            case "val":
                self.corpus_path = config.val_dataset
        self.encoding = config.encoding

        with open(self.corpus_path, "r", encoding=config.encoding) as f:
            if self.corpus_lines is None and not config.on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=config.corpus_lines):
                    self.corpus_lines += 1

            if config.on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=config.corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not config.on_memory:
            self.file = open(self.corpus_path, "r", encoding=config.encoding)
            self.random_file = open(self.corpus_path, "r", encoding=config.encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

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
        return DataLoader(self, batch_size=self.config.batch_size, num_workers=self.config.num_workers)



    @staticmethod
    def extend_parser(parser) -> argparse.ArgumentParser:
        parser.add_argument('--seq_len', type=int, default=20, help="maximum sequence length")
        parser.add_argument('--train_dataset', type=str, default='data/wikitext2/wiki.train.tokens')
        parser.add_argument('--test_dataset', type=str, default='data/wikitext2/wiki.test.tokens')
        parser.add_argument('--val_dataset', type=str, default='data/wikitext2/wiki.test.tokens')
        parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
        parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
        parser.add_argument("--encoding", type=str, default='utf-8', help="text data encoding")
        parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

        return parser
