root_dir = "/content/gdrive/My Drive/"
base_dir_read = root_dir + 'ybshmmlchk/zatoboj/SQUAD/'
base_dir_write = root_dir + 'ybshmmlchk/'

def get_tokenizer(model_conf):
    '''
    Return tokenizer based on configuration.
    '''
    if model_conf['transformer'] == 'albert':
        from transformers import AlbertTokenizer as tokenizer
    elif model_conf['transformer'] == 'bert':
        from transformers import BertTokenizer as tokenizer
    else:
        raise ValueError('Unknown transformer model. Please, select a different model or include this model in the code.')
    try:
        return tokenizer.from_pretrained(model_conf['transformer'] + '-' + model_conf['transformer_version'])
    except:
        raise ValueError(f"Unknown transformer model version: {conf['transformer'] + '-' + conf['transformer_version']}. Please, select a different transformer model version or include this version in the code.")

def get_transformer(model_conf):
    '''
    Return transformer model based on configuration.
    '''
    if model_conf['transformer'] == 'albert':
        from transformers import AlbertModel as transformer
    elif model_conf['transformer'] == 'bert':
        from transformers import BertModel as transformer
    else:
        raise ValueError('Unknown transformer model. Please, select a different model or include this model in the code.')
    try:
        return transformer.from_pretrained(model_conf['transformer'] + '-' + model_conf['transformer_version'])
    except:
        raise ValueError('Unknown transformer model version. Please, select a different transformer model version or include this version in the code.')
        
