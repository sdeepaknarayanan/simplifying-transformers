from config.train_config import RetrainMergeConfig
from parent_bert import get_pretrained_berd
import logging

import torch.backends.cudnn

import datasets
import models

torch.backends.cudnn.benchmark = True
import time
import torch.nn as nn
import torch

class combined_module(nn.Module):
    """Edit This Block for Odd Layers as needed!!!"""
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.block_model_0 = models.get(model_name=conf.model)(config=conf)
        self.block_model_1 = models.get(model_name=conf.model)(config=conf)
        self.block_model_2 = models.get(model_name=conf.model)(config=conf)
        self.block_model_3 = models.get(model_name=conf.model)(config=conf)
        self.block_model_4 = models.get(model_name=conf.model)(config=conf)
        self.block_model_5 = models.get(model_name=conf.model)(config=conf)
        self.mask_lm = models.bert.MaskedLanguageModel(conf.block_hidden_features, vocab_size = 30522)
        
    def _init(self, path):
        
        if self.conf.device == 'cpu':
            best_model_0 = torch.load(path+"/block_0", map_location=torch.device('cpu'))
            best_model_1 = torch.load(path+"/block_2", map_location=torch.device('cpu'))
            best_model_2 = torch.load(path+"/block_4", map_location=torch.device('cpu'))
            best_model_3 = torch.load(path+"/block_6", map_location=torch.device('cpu'))
            best_model_4 = torch.load(path+"/block_8", map_location=torch.device('cpu'))
            best_model_5 = torch.load(path+"/block_10", map_location=torch.device('cpu'))
        else:
            best_model_0 = torch.load(path+"/block_0")
            best_model_1 = torch.load(path+"/block_2")
            best_model_2 = torch.load(path+"/block_4")
            best_model_3 = torch.load(path+"/block_6")
            best_model_4 = torch.load(path+"/block_8")
            best_model_5 = torch.load(path+"/block_10")
        

        self.block_model_0.load_state_dict(best_model_0['model_state_dict'])
        self.block_model_1.load_state_dict(best_model_1['model_state_dict'])
        self.block_model_2.load_state_dict(best_model_2['model_state_dict'])
        self.block_model_3.load_state_dict(best_model_3['model_state_dict'])
        self.block_model_4.load_state_dict(best_model_4['model_state_dict'])
        self.block_model_5.load_state_dict(best_model_5['model_state_dict'])

        
    def forward(self, x, mask):
        
        x = self.block_model_0.forward(x, mask)
        x = self.block_model_1.forward(x, mask)
        x = self.block_model_2.forward(x, mask)
        x = self.block_model_3.forward(x, mask)
        x = self.block_model_4.forward(x, mask)
        x = self.block_model_5.forward(x, mask)
        y = self.mask_lm(x)
        return y



def train(conf):
    """
    Training for a single attention block. Similary structured as main in train.py but just for one block.

    :param conf: config object from arg parser containing all the block training configuration
    :return:

    """
    vocab = datasets.get_vocab(conf)

    train_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab)
    test_dataset = datasets.get(dataset_name=conf.dataset)(config=conf, vocab=vocab, split="test")

    train_loader = train_dataset.get_data_loader()
    test_loader = test_dataset.get_data_loader()

    base_model = get_pretrained_berd()
    
    new_model = combined_module(conf)
    
    new_model.to('cuda')
    
    new_model._init("merge_models_best")
        
    new_model.epoch = 0
    
    base_model.eval()

    logging.log(logging.INFO, "Initialized")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    epoch = new_model.epoch
    
    optimizer = torch.optim.Adam(new_model.parameters(), lr = conf.lr)
    
    # main training loop
    while epoch < conf.epochs + 1:
        
        for index, data in enumerate(train_loader):
            
            x = None
            y = None
            teacher_output = None
                        

            x = data['bert_input'].to(conf.device)

            segment_info = data['segment_label'].to(conf.device)

            teacher_output = base_model(x, segment_info)
            teacher_output = teacher_output[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            teacher_output = torch.nn.functional.softmax(teacher_output, dim=1)
            
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            x = base_model.bert.embedding(x, segment_info)
            
            optimizer.zero_grad()

            preds = new_model.forward(x, mask)
            
            y = preds[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            
            loss = criterion(y, teacher_output)
            loss.backward()
            optimizer.step()
            
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}]. Current loss: {l:.5f}'.format(
                e=epoch, b=index + 1, t=len(train_loader), l=loss))
    
            del x, mask, segment_info, loss
        
        if epoch % 5 == 0:
            import os
            if os.path.exists(conf.storage_directory):
                torch.save(new_model.state_dict(), conf.storage_directory + f"/combined_model_{epoch}.pth")
            else:
                os.makedirs(conf.storage_directory)
                torch.save(new_model.state_dict(), conf.storage_directory + f"/combined_model_{epoch}.pth")

        torch.cuda.empty_cache()
        # Evaluate on test-set.
        for index, data in enumerate(test_loader):

            x = None
            y = None
            teacher_output = None
            with torch.no_grad():
                x = data['bert_input'].to(conf.device)

                segment_info = data['segment_label'].to(conf.device)
                teacher_output = base_model(x, segment_info)
                teacher_output = teacher_output[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
                teacher_output = torch.nn.functional.softmax(teacher_output, dim=1)

                mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
                x = base_model.bert.embedding(x, segment_info)
                preds = new_model.forward(x, mask)
                y = preds[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
                loss = criterion(y, teacher_output)
            print('Epoch {e:>2}, Batch [{b:>5}/{t:<5}] eval Current loss: {l:.5f}'.format(e=epoch, b=index + 1,      t=len(test_loader), l=loss))

        del x, mask, segment_info, loss
        torch.cuda.empty_cache()
        # Write the accumulated losses to logs and reset and reset the accumulation
        # criterion.complete_epoch(epoch=epoch, mode='test')

        # Save sample images containing prediction and label side by side
        if conf.print_samples:

            data = next(iter(test_loader))

            x = data['bert_input'].to(conf.device)
            segment_info = data['segment_label'].to(conf.device)
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
            x = base_model.bert.embedding(x, segment_info)
            y = new_model.forward(x, mask)

            teacher_pred = base_model(data['bert_input'].to(conf.device), data['segment_label'].to(config.device))

            masked_prediction = y[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            predicted_label = torch.argmax(masked_prediction, dim=1)

            masked_teacher_pred = teacher_pred[torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            masked_teacher_pred = torch.argmax(masked_teacher_pred, dim=1)

            gt_label = data['bert_label'][torch.arange(data['bert_input'].size(0)), data['mask_index'].long()]
            for i in range(data['bert_input'].size(0)):
                print('GT: {g:>15}, TEACHER: {t:>15}, PRED: {p:>15}'.format(
                    g=vocab.itos[gt_label[i]],
                    t=vocab.itos[masked_teacher_pred[i]],
                    p=vocab.itos[predicted_label[i]]),
                    vocab.from_seq(data['bert_input'][i], join=True)
                )
            del gt_label, masked_teacher_pred, predicted_label, masked_prediction, teacher_pred, y, x, segment_info, mask

        torch.cuda.empty_cache()
        epoch += 1


if __name__ == "__main__":
    # gather necessary config data from different parsers and combine.
    option = RetrainMergeConfig()
    option.print()
    config = option.get()

    # start training
    train(config)
