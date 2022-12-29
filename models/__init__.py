from models.bert import BERT, BERTLM
from models.block_model import BLOCK


def get(model_name: str):
    match model_name:
        case 'BERT':
            return BERT
        case 'BERTLM':
            return BERTLM
        case 'BLOCK':
            return BLOCK
        case _:
            raise NotImplementedError

