from datasets.vocabulary import WordVocab, BertVocab
from datasets.wikitext2 import WikiText2Dataset


def get(dataset_name: str):
    match dataset_name:
        case 'wikitext2':
            return WikiText2Dataset
        case _:
            raise NotImplementedError


def get_vocab(config):
    match config.vocab:
        case 'codertimo':
            return WordVocab(config).load_vocab(config.vocab_path)
        case 'bert-google':
            return BertVocab(config)
