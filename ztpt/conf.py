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

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            d[key] = value.__dict__
        return d

def default_config():

    config_dict = {
    'dirs' : ConfigNamespace(),
    'transformer' : ConfigNamespace(),
    'model' : ConfigNamespace(),
    'train' : ConfigNamespace(),
    }

    config = ConfigNamespace(**config_dict)

    config.dirs.root = '/content/gdrive/My Drive/ybshmmlchk/'  # directories configuration
    config.dirs.data = '/content/gdrive/My Drive/ybshmmlchk/datasets/'
    config.dirs.saved_models = '/content/gdrive/My Drive/ybshmmlchk/saved_models/'
    config.dirs.logs = '/content/gdrive/My Drive/ybshmmlchk/logs/'
    config.dirs.py_drive = False

    config.transformer.model = 'albert'
    config.transformer.version = 'base-v2'

    config.model.model = 'SQUADBERT'
    config.model.max_len = 256
    config.model.batch_size = 16    
    config.model.lr = 1e-05
    config.model.freeze_layers = 0
    config.model.signature = ''

    config.model.proj_dim = 20
    config.model.weight = 1.
    config.model.answer_punishment_coeff = 1.
    config.model.num_inner_products = 5

    config.train.max_epochs = 1
    config.train.val_check_interval = 0.2
    config.train.limit_train_batches = 1.
    config.train.limit_val_batches = 1.
    config.train.accumulate_grad_batches = 1

    return config