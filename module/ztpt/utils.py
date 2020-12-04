root_dir = "/content/gdrive/My Drive/"
base_dir_read = root_dir + 'ybshmmlchk/zatoboj/SQUAD/'
base_dir_write = root_dir + 'ybshmmlchk/'

def get_tokenizer(conf):
    '''
    Return tokenizer based on configuration.
    '''
    print('hello!')
    if conf['model'] == 'albert':
        from transformers import AlbertTokenizer as tokenizer
    elif conf['model'] == 'bert':
        from transformers import BertTokenizer as tokenizer
    else:
        raise ValueError('Unknown model. Please, select a different model or include this model in the code.')
    try:
        return tokenizer.from_pretrained(conf['model'] + '-' + conf['model_version'])
    except:
        raise ValueError('Unknown model version. Please, select a different transformer model version or include this version in the code.')

def get_transformer(conf):
    '''
    Return transformer model based on configuration.
    '''
    if conf['model'] == 'albert':
        from transformers import AlbertModel as transformer
    elif conf['model'] == 'bert':
        from transformers import BertModel as transformer
    else:
        raise ValueError('Unknown model. Please, select a different model or include this model in the code.')
    try:
        return transformer.from_pretrained(conf['model'] + '-' + conf['model_version'])
    except:
        raise ValueError('Unknown model version. Please, select a different transformer model version or include this version in the code.')
        
