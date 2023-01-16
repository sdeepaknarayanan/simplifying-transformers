from models.bert import BERT, BERTLM
from models.block_model import BLOCK
from models.merged_retrained_bert import MergedRetrainedBert


def get(model_name: str):
    if model_name == 'BERT':
        return BERT
    if model_name == 'BERTLM':
        return BERTLM
    if model_name == 'BLOCK':
        return BLOCK
    if model_name == "MergedRetrainedBert":
        return MergedRetrainedBert
    else:
        raise NotImplementedError


