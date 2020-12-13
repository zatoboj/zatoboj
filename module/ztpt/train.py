import os
import pickle
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# import wandb

from .model import create_model, SQUADBERT

def load_model(model_conf=None): 
    print('hey, man!')
    print('uf')
    print('oh wow!') 
    model_save_dir = model_conf['model_save_dir']
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a good boy, copy shared saved models folder into root directory.")
    model_name = '_'.join([model_conf['transformer'],
                   model_conf['transformer_version'],
                   model_conf['model_class'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers'])
                   ])
    model_path = model_save_dir + model_name + '/model.ckpt'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if os.path.exists(model_path):
        hparams = {
            'batch_size' : model_conf['batch_size'],
            'max_len' : model_conf['max_len'],
            'freeze_layers' : model_conf['freeze_layers'],
            'lr' : model_conf['lr'],
            'model_conf' : model_conf          
        }
        model = globals()[model_conf['model_class']].load_from_checkpoint(model_path, **hparams)
        # model = create_model(model_conf, loading=True)
        # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f'Model {model_name} does not exist.')
    return model

def train_model(model_conf):
    '''
    Create, train and save model using callback.
    '''
    model_save_dir = model_conf['model_save_dir']
    if not os.path.exists(model_save_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a saved models folder ('saved_models'). Be a good boy, copy shared saved models folder into root directory.")
    model_name = '_'.join([model_conf['transformer'],
                   model_conf['transformer_version'],
                   model_conf['model_class'],
                   model_conf['unique_name'],
                   str(model_conf['max_len']),
                   str(model_conf['batch_size']),
                   str(model_conf['lr']),
                   str(model_conf['freeze_layers'])
                   ])
    model_path = model_save_dir + model_name + '/model.ckpt'
    # wandb_path = log_dir + 'wandb/' + model_name + '_id.pickle'
    
    # if not os.path.exists(log_dir):
    #     raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a logs folder ('logs'). Be a good boy, copy shared logs folder into root directory.")    

    # initializing model
    if os.path.exists(model_path):
        print('Model already exists, loading trained model...')
        model = load_model(model_conf)
        resume_from_checkpoint = model_path       
        # with open(wandb_path, 'r') as f:
        #     id = pickle.load(f)
        print('Succesfully loaded trained model. Continuing training...')
    else:
        print('Creating new model...')
        model = create_model(model_conf)
        resume_from_checkpoint = None
        # id = wandb.util.generate_id()
        # with open(wandb_path, 'w') as f:
        #     pickle.dump(id, f)
        print('Succesfully created new model. Beginning training...')
    
    # saving checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath = model_save_dir + model_name,
        filename = 'model',
        save_top_k = 1,
        verbose = True,
        monitor = 'val_loss',
        mode = 'min'
    )
    # tensorboard logger used by trainer
    log_dir = model_conf['log_dir']
    tb_logger = TensorBoardLogger(
        save_dir = log_dir + 'tensorboard',
        name = model_name,
        version = 1
    )
    # weight&biases logger used by trainer
    wandb_logger = WandbLogger(
        save_dir = model_save_dir + model_name,
        name = model_name,
        project = 'ztpt',
        # id = model_name
    )
    # training
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    trainer = pl.Trainer(
        gpus = 1, 
        amp_level = 'O2', 
        precision = 16, 
        max_epochs = model_conf['epochs'], 
        val_check_interval = model_conf['val_check_interval'], 
        checkpoint_callback = checkpoint_callback, 
        progress_bar_refresh_rate = 25,
        logger = [tb_logger, wandb_logger],
        resume_from_checkpoint = resume_from_checkpoint,
        limit_train_batches = model_conf['limit_train_batches'],
        limit_val_batches = model_conf['limit_val_batches']
        ) 
    # trainer.save_checkpoint('EarlyStoppingADam-32-0.001.pth')
    # wandb.save('EarlyStoppingADam-32-0.001.pth')   
    trainer.fit(model) 
    print('Finished training.')



