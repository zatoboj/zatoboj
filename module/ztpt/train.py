import os
import pickle, yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# import wandb
from .conf import default_config
from .model import create_model, SQUADBERT


def load_model(config = None, from_list = True): 
    if from_list:
        config = default_config()
        model_save_dir = config.dirs.saved_models
        models = os.listdir(model_save_dir)
        menu = zip([f'[{i+1}]' for i in range(len(models))], models)
        menu = [number + ' - ' + model for number, model in menu]
        menu = '\n'.join(menu)
        number = input(f'Enter number of model to load:\n{menu}\n')
        model_name = models[int(number)-1]
    else:
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
    model_path_v0 = model_save_dir + model_name + '/model-v0.ckpt'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if os.path.exists(model_path_v0):
        model_path = model_path_v0
    if os.path.exists(model_path):
        with open(model_save_dir + model_name + '/config.yaml', 'r') as f:  
            config = yaml.load(f, Loader=yaml.FullLoader) 
        hparams = {
            'batch_size' : config.model.batch_size,
            'max_len' : config.model.max_len,
            'freeze_layers' : config.model.freeze_layers,
            'lr' : config.model.lr,
            'wrapped_config' : [config]      
            }
        model = globals()[config.model.model].load_from_checkpoint(model_path, **hparams)
        # model = create_model(config, loading=True)
        # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(f'Model {model_path} does not exist.')
    return model

def train_model(config):
    '''
    Create, train and save model using callback.
    '''
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
    # wandb_path = log_dir + 'wandb/' + model_name + '_id.pickle'  

    # initializing model
    if os.path.exists(model_path):
        print('Model already exists, loading trained model...')
        model = load_model(config, from_list = False)
        resume_from_checkpoint = model_path       
        # with open(wandb_path, 'r') as f:
        #     id = pickle.load(f)
        print('Succesfully loaded trained model. Continuing training...')
    else:
        print('Creating new model...')
        model = create_model(config)
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
    log_dir = config.dirs.logs
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
        checkpoint_callback = checkpoint_callback, 
        progress_bar_refresh_rate = 25,
        logger = [tb_logger, wandb_logger],
        resume_from_checkpoint = resume_from_checkpoint,
        **config.train.__dict__
        ) 
    # trainer.save_checkpoint('EarlyStoppingADam-32-0.001.pth')
    # wandb.save('EarlyStoppingADam-32-0.001.pth')   
    trainer.fit(model) 
    with open(model_save_dir + model_name + '/config.yaml', 'w') as f:  
        yaml.dump(config, f)   
    print('Finished training.')



