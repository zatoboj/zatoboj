import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .model import create_model

def load_model(model_conf=None, train_conf=None):
    
    model_save_dir = train_conf['model_save_dir']
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a good boy, copy shared saved models folder into root directory.")
    model_name = '_'.join([model_conf['transformer'],
                   model_conf['transformer_version'],
                   model_conf['model'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers']),
                   str(train_conf['epochs'])])
    model_path = model_save_dir + model_name + '.ckpt'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if os.path.exists(model_path):
        model = create_model(model_conf)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f'Model {model_name} does not exist.')
    return model

def train_model(model_conf, train_conf):
    '''
    Create, train and save model using callback.
    '''
    model_save_dir = train_conf['model_save_dir']
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a good boy, copy shared saved models folder into root directory.")
    model_name = '_'.join([model_conf['transformer'],
                   model_conf['transformer_version'],
                   model_conf['model'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers']),
                   str(train_conf['epochs'])])
    model_path = model_save_dir + model_name + '.ckpt'
    log_dir = train_conf['log_dir']
    if not os.path.exists(log_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a logs folder ('logs'). Be a good boy, copy shared logs folder into root directory.")    
    epochs = train_conf['epochs']
    if os.path.exists(model_path):
        print('Model already exists, loading trained model...')
        model = load_model(model_conf, train_conf)
        print('Succesfully loaded trained model.')
    else:
        print('Creating new model...')
        model = create_model(model_conf)
        print('Succesfully created new model.')
    # saving checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath = model_save_dir,
        filename = model_name,
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
    )
    # tensorboard logger used by trainer
    tb_logger = TensorBoardLogger(
        save_dir = log_dir + 'tensorboard',
        name = model_name,
        version = 1
    )

    # weight&biases logger used by trainer
    wandb_logger = WandbLogger(
        save_dir = log_dir + 'weights_and_biases',
        name = model_name,
        project = 'ztpt',
        version = 1
    )

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Beginning training...')
    trainer = pl.Trainer(
        gpus = 1, 
        amp_level = 'O2', 
        precision = 16, 
        max_epochs = epochs, 
        val_check_interval = 0.1, 
        checkpoint_callback = checkpoint_callback, 
        progress_bar_refresh_rate = 25,
        logger = [tb_logger,wandb_logger]) # accelerator='ddp'   
    trainer.fit(model) 
    print('Finished training.')



