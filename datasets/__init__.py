from datasets.vocabulary import WordVocab, BertVocab
from datasets.wikitext2 import WikiText2Dataset
from datasets.BlockDataset import BlockDataset


def get(dataset_name: str):
    if dataset_name == 'wikitext2':
        return WikiText2Dataset
    if dataset_name == 'block_training':
            return BlockDataset
    else:
        return NotImplementedError


def get_vocab(config):
    if config.vocab == 'codertimo':
        return WordVocab(config).load_vocab(config.vocab_path)
    if config.vocab == 'bert-google':
        return BertVocab(config)

