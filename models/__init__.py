from models.bert import BERT, BERTLM
from models.block_model import BLOCK
from models.merge_model import MERGE


def get(model_name: str):
    if model_name == 'BERT':
        return BERT
    if model_name == 'BERTLM':
        return  BERTLM
    if model_name == 'BLOCK':
        return BLOCK
    if model_name == 'MERGE':
        return MERGE
    else:
        raise NotImplementedError

