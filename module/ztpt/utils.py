def get_tokenizer(config):
    '''
    Return tokenizer based on configuration.
    '''
    if config.transformer.model == 'albert':
        from transformers import AlbertTokenizer as tokenizer
    elif config.transformer.model == 'bert':
        from transformers import BertTokenizer as tokenizer
    else:
        raise ValueError('Unknown transformer model. Please, select a different model or include this model in the code.')
    try:
        return tokenizer.from_pretrained(config.transformer.model + '-' + config.transformer.version)
    except:
        raise ValueError(f"Unknown transformer model version: {conf['transformer'] + '-' + conf['transformer_version']}. Please, select a different transformer model version or include this version in the code.")

def get_transformer(config):
    '''
    Return transformer model based on configuration.
    '''
    if config.transformer.model == 'albert':
        from transformers import AlbertModel as transformer
    elif config.transformer.model == 'bert':
        from transformers import BertModel as transformer
    else:
        raise ValueError('Unknown transformer model. Please, select a different model or include this model in the code.')
    try:
        return transformer.from_pretrained(config.transformer.model + '-' + config.transformer.version)
    except:
        raise ValueError('Unknown transformer model version. Please, select a different transformer model version or include this version in the code.')
        
