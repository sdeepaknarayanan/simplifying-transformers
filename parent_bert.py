import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append("../")
from models.bert import BERTLM

class config():
    def __init__(self):
        self.vocab = "bert-google"
        self.vocab_path = "data/wikitext2/all.txt"
        self.bert_google_vocab = "data/uncased_L-12_H-768_A-12/vocab.txt"
        self.vocab_max_size = None
        self.vocab_min_frequency = 1
        self.dataset = "wikitext2"
        self.seq_len = 40
        self.on_memory = True
        self.corpus_lines = None
        self.train_dataset = "data/wikitext2/all.txt"
        self.encoding = "utf-8"
        self.batch_size = 1
        self.num_workers = 1
        self.hidden_features = 768
        self.layers = 12
        self.heads = 12
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = 0.1
        self.train = True
        self.lr = 1e-3
        self.adam_beta1=0.999
        self.adam_beta2=0.999
        self.adam_weight_decay = 0.01
        self.warmup_steps =1000
        self.storage_directory = "/Users/raphaelwinkler/PycharmProjects/simplifying-transformers"
        self.model = "BERTLM"


def get_pretrained_berd():

    conf = config()

    bert_ml = BERTLM(conf, 30522)

    pt_model = torch.load("torch_dump_model")

    mlm_rel_params = {}
    for name, param in pt_model.items():
        if "pooler" in name or "seq_relationship" in name:
            continue
        else:
            mlm_rel_params[name] = param


    from copy import deepcopy

    dic = deepcopy(bert_ml.state_dict())

    dic['bert.embedding.position.pe'][0] = deepcopy(pt_model['bert.embeddings.position_embeddings.weight'])

    dic['bert.embedding.token.weight'] = deepcopy(pt_model['bert.embeddings.word_embeddings.weight'])

    dic['bert.embedding.segment.weight'] = deepcopy(pt_model['bert.embeddings.token_type_embeddings.weight'])

    dic['bert.embedding.layer_norm.a_2'] = deepcopy(pt_model['bert.embeddings.LayerNorm.weight'])

    dic['bert.embedding.layer_norm.b_2'] = deepcopy(pt_model['bert.embeddings.LayerNorm.bias'])

    mapping = {
        'attention.self.query.weight':'attention.linear_layers.0.weight',
        'attention.self.query.bias':'attention.linear_layers.0.bias',
        'attention.self.key.weight':'attention.linear_layers.1.weight',
        'attention.self.key.bias':'attention.linear_layers.1.bias',
        'attention.self.value.weight':'attention.linear_layers.2.weight',
        'attention.self.value.bias':'attention.linear_layers.2.bias',
        'attention.output.dense.weight':'attention.output_linear.weight',
        'attention.output.dense.bias':'attention.output_linear.bias',
        'attention.output.LayerNorm.weight':'input_sublayer.norm.a_2',
        'attention.output.LayerNorm.bias': 'input_sublayer.norm.b_2',
        'intermediate.dense.weight':'feed_forward.w_1.weight',
        'intermediate.dense.bias':'feed_forward.w_1.bias',
        'output.dense.weight':'feed_forward.w_2.weight',
        'output.dense.bias':'feed_forward.w_2.bias',
        'output.LayerNorm.weight':'output_sublayer.norm.a_2',
        'output.LayerNorm.bias':'output_sublayer.norm.b_2',
    }

    inv_mapping = {}
    for key, value in mapping.items():
        inv_mapping[value] = key

    bert_ml.state_dict()

    cnt = 0
    for layer in range(12):
        # We have 12 transformer layers, iterate through them one by one
        for name, p_val in bert_ml.bert.transformer_blocks[layer].named_parameters():
            to_copy = f'bert.encoder.layer.{layer}.' + inv_mapping[name]
            param_to_copy = deepcopy(pt_model[to_copy])
            dic[f'bert.transformer_blocks.{layer}.' + name] = param_to_copy
            assert p_val.shape == param_to_copy.shape
            cnt+=1

    dic['mask_lm.linear.weight'] = deepcopy(pt_model['cls.predictions.transform.dense.weight'])
    dic['mask_lm.linear.bias'] = deepcopy(pt_model['cls.predictions.transform.dense.bias'])
    dic['mask_lm.decoder.weight'] = deepcopy(pt_model['cls.predictions.decoder.weight'])
    dic['mask_lm.decoder.bias'] = deepcopy(pt_model['cls.predictions.decoder.bias'])
    dic['mask_lm.layer_norm.a_2'] = deepcopy(pt_model['cls.predictions.transform.LayerNorm.weight'])
    dic['mask_lm.layer_norm.b_2'] = deepcopy(pt_model['cls.predictions.transform.LayerNorm.bias'])

    bert_ml.load_state_dict(dic)
    bert_ml.eval()

    return bert_ml