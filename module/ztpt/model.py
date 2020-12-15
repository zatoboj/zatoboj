import os
import pickle, json, yaml
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from .conf import ConfigNamespace
from .utils import get_transformer
from .preprocessing import load_data
from .val import evaluate_on_batch

def create_model(config, loading = False):
    model_save_dir = config.dirs.saved_models
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a dawg, copy shared saved models folder into root directory.")
    model_name = '_'.join([config.transformer.model,
                   config.transformer.version,
                   config.model.model,
                   config.model.signature,
                   str(config.model.max_len),
                   str(config.model.batch_size),
                   str(config.model.lr),
                   str(config.model.freeze_layers)
                   ])
    model_path = model_save_dir + model_name + '/model.ckpt'
    if not os.path.exists(model_path) or loading: 
        if not os.path.exists(model_save_dir + model_name): 
            os.mkdir(model_save_dir + model_name)
        with open(model_save_dir + model_name + '/config.yaml', 'w') as f:  
            yaml.dump(config, f)   
        batch_size = config.model.batch_size   
        max_len = config.model.max_len
        freeze_layers = config.model.freeze_layers
        lr = config.model.lr
        models_dict = {'SQUADBERT' : SQUADBERT, 'TensorBERT' : TensorBERT}
        model_class = models_dict[config.model.model]
        model = model_class([config])
        return model
    else:
        raise ValueError("Model with these configuration already exists. Please, work with the existing model, or change configuration. For example, you can add unique model signature: assign value like 'yourname-v1' to config.model.signature.")

class SQUADBERT(pl.LightningModule):
    def __init__(self, batch_size, max_len, freeze_layers, lr, config):
        super(SQUADBERT, self).__init__()    
        # initializing parameters
        self.config = config[0]
        self.batch_size = batch_size     
        self.max_len = max_len
        self.freeze_layers = freeze_layers
        self.lr = lr
        # initializing BERT
        self.bert = get_transformer(self.config).cuda()
        self.bert_dim = self.bert.config.hidden_size
        # initializing dataloaders
        self.squad_train_dataloader, self.squad_val_dataloader, self.squad_test_dataloader = generate_squad_dataloaders(self.config)
        # initializing additional layers -- start and end vectors
        self.Start = nn.Linear(self.bert_dim, 1)
        self.End = nn.Linear(self.bert_dim, 1)
        # save hyperparameters for .hparams attribute
        self.save_hyperparameters()

    def new_layers(self, bert_output, new_layer):
        logits_wrong_shape = new_layer(torch.reshape(bert_output, (bert_output.shape[0]*bert_output.shape[1], bert_output.shape[2])))
        logits = torch.reshape(logits_wrong_shape, (bert_output.shape[0], bert_output.shape[1]))
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids):
        #apply BERT
        # _ should be used for classification answer/no answer
        bert_output, _ = self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids)
        # shape of q will be (batch_size, max_len, bert_dim) = (batch_size, 256, 768)
        # take inner products of output vectors with trainable start and end vectors
        start_logits = self.new_layers(bert_output, self.Start)
        end_logits = self.new_layers(bert_output, self.End)

        return start_logits, end_logits

    # this is the main function of pl modules. defines architecture and loss function. training loop comes for free -- implemented inside PL
    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch
         
        # fwd
        start_logits, end_logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        # LOSS

        # compute cross_entropy loss between predictions and actual labels for start and end 
        start_loss = F.cross_entropy(start_logits, answer_starts)
        end_loss = F.cross_entropy(end_logits, answer_ends)
        loss = start_loss + end_loss

        # logs
        self.log('train_loss', loss, prog_bar=True)
        self.log('start_loss', start_loss, prog_bar=True)
        self.log('end_loss', end_loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch

        # fwd
        start_logits, end_logits = self.forward(input_ids, attention_mask, token_type_ids)

        # loss

        loss1 = F.cross_entropy(start_logits, answer_starts)
        loss2 = F.cross_entropy(end_logits, answer_ends)
        loss = loss1 + loss2

        # ^^^^ the code above is the same as for training step, but we also add accuracy computation for validation below

        _, accuracy_dict = evaluate_on_batch(self, batch, metrics = ['plain','bysum','byend'])
    
        # logs
        self.log('val_loss', loss, prog_bar=True)
        
        return accuracy_dict

    def validation_epoch_end(self, val_step_outputs):
        log_dict = {}
        for key in val_step_outputs[0]:
            aggregated = np.mean([accuracy_dict[key] for accuracy_dict in val_step_outputs])
            log_dict[key] = aggregated
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader

    def test_dataloader(self):
        return self.squad_test_dataloader

class TensorBERT(pl.LightningModule):

    def __init__(self, wrapped_config):#proj_dim, num_inner_products, batch_size, weight =2., answer_punishment_coeff=1.):
        self.config = wrapped_config[0]
        super(TensorBERT, self).__init__() 
        self.bert = get_transformer(self.config).cuda()
        self.bert_dim = self.bert.config.hidden_size #768
        self.weight = self.config.model.weight
        self.answer_punishment_coeff = self.config.model.answer_punishment_coeff
        self.batch_size = self.config.model.batch_size
        self.max_len = self.config.model.max_len
        self.proj_dim = self.config.model.proj_dim
        self.Proj = nn.Linear(self.bert_dim, self.proj_dim)
        self.Proj_cls = nn.Linear(self.bert_dim, self.proj_dim)
        self.BL = nn.Bilinear(self.proj_dim, self.proj_dim, self.config.model.num_inner_products) # l scalar products of 2 vectors of dim d
        self.L = nn.Linear(self.config.model.num_inner_products, 2)
        self.CLS = nn.Linear(self.bert_dim, 2) #(a,b) e^a/(e^a+e^b)
        self.squad_train_dataloader, self.squad_val_dataloader, self.squad_test_dataloader = generate_squad_dataloaders(self.config)
        self.save_hyperparameters()

    def my_forward_pass(self, cls_bert_output, bert_output_full):
        current_batch_size = bert_output_full.shape[0]
        bert_output_full = torch.reshape(bert_output_full, (current_batch_size * self.max_len, self.bert_dim))
        proj_output_full = self.Proj(bert_output_full)
        proj_cls = self.Proj_cls(cls_bert_output)
        proj_cls = torch.cat([proj_cls]*self.max_len) # replicated proj_cls to make it the same shape as proj_output_full
        long_logits = self.BL(proj_cls, proj_output_full)
        long_logits = nn.ReLU6()(long_logits)
        long_logits = self.L(long_logits)
        long_logits = torch.reshape(long_logits, (current_batch_size, self.max_len, 2))
        return long_logits

    def forward(self, input_ids, attention_mask, token_type_ids):
      
        bert_output_full, cls_pooler_output = self.bert(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         token_type_ids=token_type_ids)
        # bert_output_full.shape = (batch_size, max_len, bert_dim) -- one vector of dim=bert_dim for each token
        # cls_pooler_output of shape (batch_size, bert_dim) -- Last layer hidden-state of the first token of the sequence (classification token) 
        # further processed by a Linear layer and a Tanh activation function. 
        # The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.

        cls_bert_output = bert_output_full[:, 0, :] # vector corresponding to CLS token

        # long_logits will have shape (batch_size, max_len, 2)
        # each output of bert is projected to smaller dimension, then take a few inner products with projection of the cls vector,
        # then another dense layer to get logits
        long_logits = self.my_forward_pass(cls_bert_output, bert_output_full)
        cls_logits = self.CLS(cls_pooler_output)

        return cls_logits, long_logits

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch
         
        # fwd
        cls_logits, long_logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        # loss for not guessing if there is an answer
        loss1 = F.cross_entropy(cls_logits, label, weight = torch.Tensor([2.,1.]))

        # loss for each individual word -- is it in the answer?
        # TODO: need to insert weight -- around 90 bc of mismatch of 0s and 1s -- only 1% are 1s
        loss2 = F.cross_entropy(torch.reshape(long_logits, (long_logits.shape[0] * long_logits.shape[1], long_logits.shape[2])), 
                                torch.reshape(answer_mask, (answer_mask.shape[0] * answer_mask.shape[1],)), weight = torch.Tensor([1.,50.]))
        # total loss
        # TODO: experiment with punishment coeff
        loss = self.answer_punishment_coeff*loss1 + loss2
        self.log('train_loss', loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = batch
         
        # fwd
        cls_logits, long_logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        # loss for not guessing if there is an answer
        loss1 = F.cross_entropy(cls_logits, label, weight = torch.Tensor([2.,1.]))

        # loss for each individual word -- is it in the answer?
        # TODO: need to insert weight -- around 90 bc of mismatch of 0s and 1s -- only 1% are 1s
        loss2 = F.cross_entropy(torch.reshape(long_logits, (long_logits.shape[0] * long_logits.shape[1], long_logits.shape[2])), 
                                torch.reshape(answer_mask, (answer_mask.shape[0] * answer_mask.shape[1],)))#, weight = torch.Tensor([1.,self.weight]))
        # total loss
        # TODO: experiment with punishment coeff
        val_loss = self.answer_punishment_coeff*loss1 + loss2

        # compute accuracy 
        # TODO: compute precision/recall on individual words, accuracy of start, end, exact match

        # ну хоть одна переменная должна нормально называться
        _ , y1 = torch.max(cls_logits, dim=1)
        label_acc = torch.sum(y1 == label) / label.shape[0]
        #self.log('label_acc', label_acc, prog_bar=True)
        #self.log('val_loss', val_loss, prog_bar=True)

        return {'val_loss' : val_loss, 'label_acc' : label_acc}
        
    def dics_average(self, dics, name):
        d = dict()
        for key in dics[0].keys():
          d[name + key] = torch.stack([x[key] for x in dics]).mean()
        return d
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

        label_acc = torch.stack([x['label_acc'] for x in outputs]).mean()
        self.log('val_label_acc', label_acc, prog_bar=True)


    '''
    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        
        return {'test_acc': torch.tensor(test_acc)}

    def test_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
    '''
    
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=5e-06, eps=1e-08)

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader
    
    def test_dataloader(self):
        return self.squad_test_dataloader


def generate_squad_dataloaders(config):
    # ----------------------
    # TRAIN/VAL/TEST DATALOADERS
    # ----------------------
    batch_size = config.model.batch_size
    train_data, val_data, test_data  = load_data(config)
    # TensorDataset from training examples. ".cuda()" puts the corresponding tensor on gpu
    squad_train_dataset = TensorDataset(torch.tensor(train_data['input_ids'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(train_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(train_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(train_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['indexing'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['answer_starts'], dtype=torch.long).cuda(),
                                torch.tensor(train_data['answer_ends'], dtype=torch.long).cuda()
                                )
    # TensorDataset from validation examples.
    squad_val_dataset = TensorDataset(torch.tensor(val_data['input_ids'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(val_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(val_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(val_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['indexing'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['answer_starts'], dtype=torch.long).cuda(),
                                torch.tensor(val_data['answer_ends'], dtype=torch.long).cuda())
  
    # TensorDataset from test examples.
    squad_test_dataset = TensorDataset(torch.tensor(test_data['input_ids'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['attention_mask'], dtype=torch.long).cuda(),  
                                torch.tensor(test_data['token_type_ids'], dtype=torch.long).cuda(), 
                                1 - torch.tensor(test_data['labels'], dtype=torch.long).cuda(), #label is 0 if there is an answer in the original dataset
                                torch.tensor(test_data['answer_mask'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['indexing'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['answer_starts'], dtype=torch.long).cuda(),
                                torch.tensor(test_data['answer_ends'], dtype=torch.long).cuda())

    # train loader
    train_sampler = RandomSampler(squad_train_dataset)
    squad_train_dataloader = DataLoader(squad_train_dataset, sampler = train_sampler, batch_size = batch_size)

    # val loader
    val_sampler = SequentialSampler(squad_val_dataset)
    squad_val_dataloader = DataLoader(squad_val_dataset, sampler = val_sampler, batch_size = batch_size, shuffle = False)

    # test loader
    test_sampler = SequentialSampler(squad_test_dataset)
    squad_test_dataloader = DataLoader(squad_test_dataset, sampler=test_sampler, batch_size = batch_size, shuffle = False)

    return squad_train_dataloader, squad_val_dataloader, squad_test_dataloader 
