import os
import pickle, json
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from transformers import AlbertModel, AlbertTokenizer
from .preprocessing import preprocess

root_dir = "/content/gdrive/My Drive/"
base_dir_read = root_dir + 'ybshmmlchk/zatoboj/SQUAD/'
base_dir_write = root_dir + 'ybshmmlchk/'

def load_data(conf):
    preprocess(conf)
    conf_prefix = conf['model'] + '-' + conf['model_version'] + '-' + str(conf['max_len']) 
    train_data_path = base_dir_write + conf_prefix + '-train.pickle'
    test_data_path = base_dir_write + conf_prefix + '-test.pickle'
    val_data_path = base_dir_write + conf_prefix + '-val.pickle'
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f)    
    print('Data has been succesfully loaded')
    print('Train data size:'.ljust(21), len(train_data['labels']))
    print('Test data size:'.ljust(21), len(test_data['labels']))
    print('Validation data size:'.ljust(21), len(val_data['labels']))                
    return train_data, test_data, val_data

def generate_squad_dataloaders(batch_size, conf):
    # ----------------------
    # TRAIN/VAL/TEST DATALOADERS
    # ----------------------
    train_data, test_data, val_data = load_data(conf)
    # TensorDataset from training examples. ".cuda()" puts the corresponding tensor on gpu
    squad_train_dataset = TensorDataset(torch.tensor(train_data['input_ids'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(train_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(train_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(train_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['indexing'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['answer_starts']).cuda(),
                                torch.tensor(train_data['answer_ends']).cuda())
    # TensorDataset from validation examples.
    squad_val_dataset = TensorDataset(torch.tensor(val_data['input_ids'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(val_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(val_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(val_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['indexing'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['answer_starts']).cuda(),
                                torch.tensor(val_data['answer_ends']).cuda())
  
    # TensorDataset from test examples.
    squad_test_dataset = TensorDataset(torch.tensor(test_data['input_id'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(test_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(test_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(test_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['indexing'], dtype=torch.long).cuda())

    # train loader
    train_sampler = RandomSampler(squad_train_dataset)
    squad_train_dataloader = DataLoader(squad_train_dataset, sampler = train_sampler, batch_size = batch_size)

    # val loader
    val_sampler = RandomSampler(squad_val_dataset)
    squad_val_dataloader = DataLoader(squad_val_dataset, sampler = val_sampler, batch_size = batch_size, shuffle = False)

    # test loader
    test_sampler = RandomSampler(squad_test_dataset)
    squad_test_dataloader = DataLoader(squad_test_dataset, sampler=test_sampler, batch_size = batch_size)

    return squad_train_dataloader, squad_test_dataloader, squad_val_dataloader

class SQUADBERT(pl.LightningModule):
    def __init__(self, batch_size, conf, freeze_layers = 0):
        super(SQUADBERT, self).__init__()    
        # initializing parameters
        self.batch_size = batch_size
        self.conf = conf
        self.max_len = conf['max_len']
        self.freeze_layers = freeze_layers
        # initializing BERT
        self.bert = bert.cuda()
        self.n = bert.config.hidden_size
        # initializing dataloaders
        self.squad_train_dataloader, self.squad_test_dataloader, self.squad_val_dataloader = generate_squad_dataloaders(self.batch_size)
        # initializing additional layers -- start and end vectors
        self.Start = nn.Linear(self.n, 1)
        self.End = nn.Linear(self.n, 1)

    def new_layers(self, q, new_layer):
        logits_wrong_shape = new_layer(torch.reshape(q, (q.shape[0]*q.shape[1], q.shape[2])))
        logits = torch.reshape(logits_wrong_shape, (q.shape[0], q.shape[1]))
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        #apply BERT
        q, _ = self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids)
        # shape of q will be (batch_size, max_len, bert_dim) = (batch_size, 256, 768)
        # take inner products of output vectors with trainable start and end vectors
        start_logits = self.new_layers(q, self.Start)
        end_logits = self.new_layers(q, self.End)

        return start_logits, end_logits

    # this is the main function of pl modules. defines architecture and loss function. training loop comes for free -- implemented inside PL
    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch
         
        # fwd
        start_logits, end_logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        # LOSS
        # get start and end positions from answer_mask
        start, end = answer_starts, answer_ends#get_start_end(answer_mask)

        # compute cross_entropy loss between predictions and actual labels for start and end 
        loss1 = F.cross_entropy(start_logits, start)
        loss2 = F.cross_entropy(end_logits, end)
        loss = loss1 + loss2

        self.log('loss', loss, prog_bar=True)
        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch

        # fwd
        start_logits, end_logits = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        start, end = answer_starts, answer_ends#get_start_end(answer_mask)

        loss1 = F.cross_entropy(start_logits, start)
        loss2 = F.cross_entropy(end_logits, end)
        loss = loss1 + loss2

        # ^^^^ the code above is the same as for training step, but we also add accuracy computation for validation below

        # acc
        a, y1 = torch.max(start_logits, dim=1)
        a, y2 = torch.max(end_logits, dim=1)
        
        start_acc = torch.sum(y1 == start) / self.batch_size
        end_acc = torch.sum(y2 == end) / self.batch_size
        self.log('start_acc', start_acc, prog_bar=True)
        self.log('end_acc', end_acc, prog_bar=True)
        return {'val_loss' : loss, 'start_acc' : start_acc, 'end_acc' : end_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        start_acc = torch.stack([x['start_acc'] for x in outputs]).mean()
        end_acc = torch.stack([x['end_acc'] for x in outputs]).mean()

        self.log('validation_loss', avg_loss, prog_bar=True)
        self.log('start_accuracy', start_acc, prog_bar=True)
        self.log('end_accuracy', end_acc, prog_bar=True)
        tensorboard_logs = {'val_loss': avg_loss, 'start_acc' : start_acc, 'end_acc' : end_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=3e-05, eps=1e-08)

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader

