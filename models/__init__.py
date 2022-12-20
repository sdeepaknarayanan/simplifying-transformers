from models.bert import BERT, BERTLM


def get(model_name: str):
    match model_name:
        case 'BERT':
            return BERT
        case 'BERTLM':
            return BERTLM
        case _:
            raise NotImplementedError
