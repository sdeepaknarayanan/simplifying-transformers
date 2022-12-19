from models.bert import BERT


def get(model_name: str):
    match model_name:
        case 'BERT':
            return BERT
        case _:
            raise NotImplementedError
