from models.bert import BERT, BERTLM
from models.block_model import BLOCK


def get(model_name: str):
    if model_name == 'BERT':
        return BERT
    if model_name == 'BERTLM':
        return  BERTLM
    if model_name == 'BLOCK':
        return BLOCK
    else:
        raise NotImplementedError

