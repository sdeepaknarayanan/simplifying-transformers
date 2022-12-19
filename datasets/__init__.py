from datasets.vocabulary import WordVocab
from datasets.wikitext2 import WikiText2Dataset


def get(dataset_name: str):
    match dataset_name:
        case 'wikitext2':
            return WikiText2Dataset
        case _:
            raise NotImplementedError


def get_vocab(config):
    return WordVocab(config)