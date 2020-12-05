import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .model import SQUADBERT

def load_model(model_conf, train_conf):
    model_save_dir = model_conf['model_save_dir']
    model_name = '_'.join([model_conf['model'],
                   model_conf['model_version'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers'])])
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = SQUADBERT(model_conf)
    if os.path.exists(model_save_dir + model_name):
        checkpoint = torch.load(base_dir + 'saved_models/' + model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('Model does not exist.')
    return model

def train_model(model_conf, train_conf):
    '''
    Create, train and save model using callback.
    '''
    model_save_dir = train_conf['model_save_dir']
    model_name = '_'.join([model_conf['model'],
                   model_conf['model_version'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers'])])
    log_dir = train_conf['log_dir']
    epochs = train_conf['epochs']
    if os.path.exists(model_save_dir + model_name):
        print('Model already exists, loading trained model...')
        model = load_model(model_conf, train_conf)
        print('Succesfully loaded trained model.')
    else:
        print('Creating new model...')
        model = SQUADBERT(model_conf)
        print('Succesfully created new model.')
    # saving checkpoint
    checkpoint_callback = ModelCheckpoint(
        filepath = model_save_dir + model_name,
        save_top_k = 2,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
    )
    # logger used by trainer
    logger = TensorBoardLogger(
        save_dir = log_dir,
        version=1,
        name='lightning_logs'
    )

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Beginning training...')
    trainer = pl.Trainer(
        gpus = 1, 
        amp_level = 'O2', 
        precision = 16, 
        max_epochs = epochs, 
        val_check_interval = 0.25, 
        checkpoint_callback = checkpoint_callback, 
        progress_bar_refresh_rate = 25,
        logger = logger) # accelerator='ddp'   
    trainer.fit(model) 
    print('Finished training.')



