class ConfigNamespace:
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return ConfigNamespace(**entry)
        return entry
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, ConfigNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)

conf_dict = {
    'dirs' : ConfigNamespace(),
    'transformer' : ConfigNamespace(),
    'model' : ConfigNamespace(),
    'train' : ConfigNamespace()
    }

config = ConfigNamespace(**conf_dict)

config.dirs.root_dir = '/content/gdrive/My Drive/ybshmmlchk/'  # directories configuration
config.dirs.data_dir = '/content/gdrive/My Drive/ybshmmlchk/datasets/'
config.dirs.model_save_dir = '/content/gdrive/My Drive/ybshmmlchk/saved_models'
config.dirs.log_dir = '/content/gdrive/My Drive/ybshmmlchk/logs/'

config.transformer.model = 'albert'
config.transformer.version = 'base-v2'

config.model.max_len = 256
config.model.batch_size = 16
config.model.model = 'SQUADBERT'
config.model.lr = 1e-05
config.model.freeze_layers = 0
config.model.unique_name = ''

config.train.epochs = 1
config.train.val_check_interval = 1.
config.train.limit_train_batches = 1.
config.train.limit_val_batches = 1.