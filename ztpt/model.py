import os
import pickle, json, yaml
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
from .conf import ConfigNamespace
from .utils import get_transformer, numpify
from .preprocessing import load_data
from .val import Evaluator

def get_model_name(config):
    '''
    Return canonical name of model based on configuration.
    '''
    model_name = '_'.join([config.transformer.model,
                   config.transformer.version,
                   config.model.model,
                   config.model.signature,
                   str(config.model.max_len),
                   str(config.model.batch_size),
                   str(f'{config.model.lr:.0e}'),
                   str(config.model.freeze_layers)
                   ])
    return model_name

def create_model(config, loading = False):
    '''
    Return new model based on configration. If loading existing model use `load_model` instead.
    '''
    model_save_dir = config.dirs.saved_models
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a dawg, copy shared saved models folder into root directory.")
    model_name = get_model_name(config)
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
    def __init__(self, wrapped_config):
        super(SQUADBERT, self).__init__()    
        # initializing parameters
        self.config = wrapped_config[0]
        self.batch_size = self.config.model.batch_size     
        self.max_len = self.config.model.max_len
        self.freeze_layers = self.config.model.freeze_layers
        self.lr = self.config.model.lr
        # save hyperparameters for .hparams attribute
        self.save_hyperparameters()
        # initializing BERT
        self.bert = get_transformer(self.config).cuda()
        self.bert_dim = self.bert.config.hidden_size
        # evaluation metrics
        self.val_metrics = ['plain', 'bysum', 'byend']
        # initializing dataloaders
        self.squad_train_dataloader, self.squad_val_dataloader, self.squad_test_dataloader = generate_squad_dataloaders(self.config)
        # initializing additional layers -- start and end vectors
        self.Start = nn.Linear(self.bert_dim, 1)
        self.End = nn.Linear(self.bert_dim, 1)
        self.custom_step = 0
        
    def new_layers(self, bert_output, new_layer):
        logits_wrong_shape = new_layer(torch.reshape(bert_output, (bert_output.shape[0]*bert_output.shape[1], bert_output.shape[2])))
        logits = torch.reshape(logits_wrong_shape, (bert_output.shape[0], bert_output.shape[1]))
        return logits

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids, _, _, _, _, _ = batch
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
        predictions = self.forward(batch)     
        loss = self.compute_loss(predictions, batch)

        self.custom_step += batch[0].shape[0]
        # logs
        self.logger.experiment.log({
            'train_loss' : loss,
            'epoch' : self.current_epoch
            }, step = self.custom_step)

        return {'loss': loss}

    def compute_loss(self, predictions, batch):
        start_logits, end_logits = predictions
        # LOSS: compute cross_entropy loss between predictions and actual labels for start and end 
        _, _, _, _, _, _, answer_starts, answer_ends = batch
        start_loss = F.cross_entropy(start_logits, answer_starts)
        end_loss = F.cross_entropy(end_logits, answer_ends)
        loss = start_loss + end_loss
        return loss

    def validation_step(self, batch, batch_nb):
        evaluator = Evaluator(self)
        _, val_dict = evaluator.evaluate_on_batch(batch) 
        return val_dict

    def validation_epoch_end(self, val_step_outputs):
        log_dict = {}
        for key in val_step_outputs[0]:
            aggregated = np.mean([accuracy_dict[key] for accuracy_dict in val_step_outputs])
            log_dict[key] = aggregated
        self.logger.experiment.log(log_dict, step = self.custom_step)
        self.log('val_loss', log_dict['val_loss'], prog_bar=True, logger=False)
        # delete models from Trash using pydrive
        if self.config.dirs.py_drive:
            for a_file in self.config.dirs.py_drive.ListFile({'q': "trashed=true"}).GetList():
                if a_file['title'] in {'model.ckpt', 'model-v0.ckpt'}:
                    a_file.Delete()
                  
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def get_predictions(self, batch):
        '''
        Returns arrays (start probabilities, end probabilities) on given batch 
        '''    
        with torch.no_grad():
            start_prob, end_prob = self.forward(batch)
        return start_prob, end_prob

    def convert_predictions(self, predictions, min_start, metric='plain'):
        '''
        Return numpy arrays of predictions of indices of starts and ends for:
        - metric='plain' - as argmax of unnormalized probability vectors
        - metric='bysum' - as argmax of the sum of unrromalized probabilities over all pairs (i,j) such that i<j (and i>min_start if given)
        - metric='byend' - as argmax of unrromalized probabilities over all i>min_start for end and
                        as argmax of unrromalized probabilities over all min_start<j<end_pred for start   
        '''
        start_prob, end_prob = predictions
        neg_inf = -100
        batch_size, max_len = start_prob.shape
        if metric == 'plain':
            start_pred = np.argmax(start_prob, axis=1)
            end_pred = np.argmax(end_prob, axis=1)       
        elif metric == 'bysum':
            probs = start_prob.reshape(-1,max_len,1) + end_prob.reshape(-1,1,max_len) # array of shape: (batch_size, max_len, max_len), matrix of pairwise sums per each element of the batch
            mask = np.zeros(probs.shape)  # create a mask to avoid including cases where i > j or i > min_start or j > min_start
            for i,s in enumerate(min_start):
                mask[i,:s,:] = 1
                mask[i,:,:s] = 1
                mask[i][np.tril_indices(max_len,-1)] = 1
            mask[:,0,0] = 0               # we however leave i=j=0 to detect questions without answers
            probs = np.ma.array(probs,mask=mask)
            probs = np.ma.filled(probs,neg_inf)
            max_probs = np.argmax(probs.reshape(batch_size,-1), axis=-1) # array of shape: (batch_size,), argmaxes of flattened matrices of pairwise sums
            start_pred, end_pred = np.unravel_index(max_probs, (max_len, max_len)) # two arrays of shape: (batch_size,), 'unflattenning' of max_probs
        elif metric == 'byend':
            # first we deal with ends
            mask = np.zeros(end_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,s in enumerate(min_start):
                mask[i,:s] = 1
            mask[:,0] = 0               # we however leave end=0 to detect questions without answers
            end_prob = np.ma.array(end_prob,mask=mask)
            start_prob = np.ma.array(start_prob,mask=mask)
            end_prob = np.ma.filled(end_prob,neg_inf)
            start_prob = np.ma.filled(start_prob,neg_inf)
            end_pred = np.argmax(end_prob, axis=-1) # array of shape: (batch_size,), argmaxes of ends' probabilities
            # now we deal with starts
            mask = np.zeros(start_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,e in enumerate(end_pred):
                mask[i,e+1:] = 1
            start_prob = np.ma.array(start_prob,mask=mask)
            start_prob = np.ma.filled(start_prob,neg_inf)
            start_pred = np.argmax(start_prob, axis=-1) # array of shape: (batch_size,), argmaxes of starts' probabilities
        return start_pred, end_pred

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader

    def test_dataloader(self):
        return self.squad_test_dataloader

class TensorBERT(pl.LightningModule):

    def __init__(self, wrapped_config):#proj_dim, num_inner_products, batch_size, weight =2., answer_punishment_coeff=1.):
        super(TensorBERT, self).__init__() 
        self.config = wrapped_config[0]
        self.batch_size = self.config.model.batch_size
        self.max_len = self.config.model.max_len
        self.proj_dim = self.config.model.proj_dim
        self.weight = self.config.model.weight
        self.answer_punishment_coeff = self.config.model.answer_punishment_coeff
        self.num_inner_products = self.config.model.num_inner_products
        self.val_metrics = ['plain']
        self.lr = self.config.model.lr

        self.bert = get_transformer(self.config).cuda()
        self.bert_dim = self.bert.config.hidden_size #768
           
        self.Proj = nn.Linear(self.bert_dim, self.proj_dim)
        self.Proj_cls = nn.Linear(self.bert_dim, self.proj_dim)
        self.BL = nn.Bilinear(self.proj_dim, self.proj_dim, self.num_inner_products) # l scalar products of 2 vectors of dim d
        self.L = nn.Linear(self.num_inner_products, 2)
        self.CLS = nn.Linear(self.bert_dim, 2) #(a,b) e^a/(e^a+e^b)
        self.squad_train_dataloader, self.squad_val_dataloader, self.squad_test_dataloader = generate_squad_dataloaders(self.config)
        self.save_hyperparameters()
        self.custom_step = 0

    def my_forward_pass(self, cls_bert_output, bert_output_full):
        current_batch_size = bert_output_full.shape[0]
        bert_output_full = torch.reshape(bert_output_full, (current_batch_size * self.max_len, self.bert_dim))
        proj_output_full = self.Proj(bert_output_full)
        proj_cls = self.Proj_cls(cls_bert_output)
        proj_cls = torch.cat([proj_cls]*self.max_len) # replicated proj_cls to make it the same shape as proj_output_full
        long_logits = self.BL(proj_cls, proj_output_full)
        long_logits = nn.LeakyReLU(negative_slope=0.1)(long_logits)
        long_logits = self.L(long_logits)
        long_logits = torch.reshape(long_logits, (current_batch_size, self.max_len, 2))
        return long_logits

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids, _, _, _, _, _ = batch        
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
        #cls_logits will have shape (batch_size, 2)
        return cls_logits, long_logits

    def training_step(self, batch, batch_nb):
        predictions = self.forward(batch)     
        loss = self.compute_loss(predictions, batch)

        self.custom_step += batch[0].shape[0]
        # logs
        self.logger.experiment.log({
            'train_loss' : loss,
            'epoch' : self.current_epoch
            }, step = self.custom_step)

        return {'loss': loss}

    def compute_loss(self, predictions, batch):
        cls_logits, long_logits = predictions
        _, _, _, label, answer_mask, _, _, _ = batch
        # loss for not guessing if there is an answer
        loss1 = F.cross_entropy(cls_logits, label, weight = torch.Tensor([self.weight,1.]))
        # loss for each individual word -- is it in the answer?
        # TODO: need to insert pass weight -- around 90? bc of mismatch of 0s and 1s -- only 1% are 1s
        loss2 = F.cross_entropy(torch.reshape(long_logits, (long_logits.shape[0] * long_logits.shape[1], long_logits.shape[2])), 
                                torch.reshape(answer_mask, (answer_mask.shape[0] * answer_mask.shape[1],)), weight = torch.Tensor([1.,50.]))
        
        loss = self.answer_punishment_coeff*loss1 + loss2
        return loss

    def validation_step(self, batch, batch_nb):
        evaluator = Evaluator(self)
        _, val_dict = evaluator.evaluate_on_batch(batch) 
        return val_dict

    def validation_epoch_end(self, val_step_outputs):
        log_dict = {}
        for key in val_step_outputs[0]:
            aggregated = np.mean([accuracy_dict[key] for accuracy_dict in val_step_outputs])
            log_dict[key] = aggregated
        self.logger.experiment.log(log_dict, step = self.custom_step)
        self.log('val_loss', log_dict['val_loss'], prog_bar=True, logger=False)
        # delete models from Trash using pydrive
        if self.config.dirs.py_drive:
            for a_file in self.config.dirs.py_drive.ListFile({'q': "trashed=true"}).GetList():
                if a_file['title'] in {'model.ckpt', 'model-v0.ckpt'}:
                    a_file.Delete()
  
    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

    def train_dataloader(self):
        return self.squad_train_dataloader

    def val_dataloader(self):
        return self.squad_val_dataloader
    
    def test_dataloader(self):
        return self.squad_test_dataloader

    def get_predictions(self, batch):
        '''
        Returns arrays (label probabilities, individual word probabilities) on given batch 
        '''    
        with torch.no_grad():
            labels_prob, individual_words_prob = self.forward(batch)
        
        return labels_prob, individual_words_prob

    def convert_predictions(self, predictions, min_start, metric='plain'):
        '''
        TODO: write proper description once a few metrics are added
        Return numpy arrays of predictions of indices of starts and ends for:
        - metric='plain' - as argmax of unnormalized probability vectors
        - metric='bysum' - as argmax of the sum of unrromalized probabilities over all pairs (i,j) such that i<j (and i>min_start if given)
        - metric='byend' - as argmax of unrromalized probabilities over all i>min_start for end and
                        as argmax of unrromalized probabilities over all min_start<j<end_pred for start   
        '''
        labels_prob, individual_words_prob = predictions
        labels_prob = labels_prob[:, 1] - labels_prob[:, 0]
        individual_words_prob = individual_words_prob[:, :, 1] - individual_words_prob[:, :, 0]
        neg_inf = -100
        batch_size, max_len = individual_words_prob.shape
        if metric == 'plain':
            labels_pred = (labels_prob>0).astype(int)

            max_indices = np.argmax(individual_words_prob, axis=1)
            start_pred = np.zeros(labels_pred.shape)
            end_pred = np.zeros(labels_pred.shape)
            for i in range(batch_size):
                if individual_words_prob[i, max_indices[i]] <=0:
                    start_pred[i] = 0
                    end_pred[i] = 0
                    continue

                current_index = max_indices[i] - 1
                while True:
                    if current_index >= min_start[i]:
                        if individual_words_prob[i, current_index] > 0:
                            current_index-=1
                        else:
                            break
                    else:
                        break
                start_pred[i] = current_index + 1

                current_index = max_indices[i] + 1
                while True:
                    if current_index < max_len:
                        if individual_words_prob[i, current_index] > 0:
                            current_index += 1
                        else:
                            break
                    else:
                        break
                start_pred[i] = current_index - 1

            start_pred = start_pred * labels_pred
            end_pred = end_pred * labels_pred

        elif metric == 'bysum':
            probs = start_prob.reshape(-1,max_len,1) + end_prob.reshape(-1,1,max_len) # array of shape: (batch_size, max_len, max_len), matrix of pairwise sums per each element of the batch
            mask = np.zeros(probs.shape)  # create a mask to avoid including cases where i > j or i > min_start or j > min_start
            for i,s in enumerate(min_start):
                mask[i,:s,:] = 1
                mask[i,:,:s] = 1
                mask[i][np.tril_indices(max_len,-1)] = 1
            mask[:,0,0] = 0               # we however leave i=j=0 to detect questions without answers
            probs = np.ma.array(probs,mask=mask)
            probs = np.ma.filled(probs,neg_inf)
            max_probs = np.argmax(probs.reshape(batch_size,-1), axis=-1) # array of shape: (batch_size,), argmaxes of flattened matrices of pairwise sums
            start_pred, end_pred = np.unravel_index(max_probs, (max_len, max_len)) # two arrays of shape: (batch_size,), 'unflattenning' of max_probs
        elif metric == 'byend':
            # first we deal with ends
            mask = np.zeros(end_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,s in enumerate(min_start):
                mask[i,:s] = 1
            mask[:,0] = 0               # we however leave end=0 to detect questions without answers
            end_prob = np.ma.array(end_prob,mask=mask)
            start_prob = np.ma.array(start_prob,mask=mask)
            end_prob = np.ma.filled(end_prob,neg_inf)
            start_prob = np.ma.filled(start_prob,neg_inf)
            end_pred = np.argmax(end_prob, axis=-1) # array of shape: (batch_size,), argmaxes of ends' probabilities
            # now we deal with starts
            mask = np.zeros(start_prob.shape)  # create a mask to avoid including cases where end > min_start
            for i,e in enumerate(end_pred):
                mask[i,e+1:] = 1
            start_prob = np.ma.array(start_prob,mask=mask)
            start_prob = np.ma.filled(start_prob,neg_inf)
            start_pred = np.argmax(start_prob, axis=-1) # array of shape: (batch_size,), argmaxes of starts' probabilities
        return start_pred, end_pred

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
