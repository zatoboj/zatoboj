import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm_notebook as tqdm

def numpify(*tensors):
    '''
    Given torch tensors as arguments return a list of corresponding numpy arrays.
    '''
    np_arrays = []
    for tensor in tensors:
        np_arrays.append(tensor.detach().cpu().numpy())
    return np_arrays

def predict(model, batch):
    '''
    Return numpy arrays of unnormalized probabilities of start and end. 
    Note that these are not actual prediction probabilities, because we didn't take softmax.
    '''
    input_ids, attention_mask, token_type_ids, _, _, _, _, _ = batch
    
    with torch.no_grad():
        start_prob, end_prob = model(input_ids, attention_mask, token_type_ids)
    start_prob, end_prob = numpify(start_prob, end_prob)
    return start_prob, end_prob

def convert_predictions_plain(start_prob, end_prob):
    '''
    Return numpy arrays of predictions of indices of starts and ends
    as argmax of unnormalized probability vectors.
    '''
    start_pred = np.argmax(start_prob, axis=1)
    end_pred = np.argmax(end_prob, axis=1)
    return start_pred, end_pred

def convert_predictions_bysum(start_prob, end_prob, min_start=None):
    '''
    Return numpy arrays of predictions of indices of starts and ends
    as argmax of the sum of unrromalized probabilities over all pairs (i,j) such that i<j (and i>min_start if given).
    '''
    neg_inf = -100
    batch_size, max_len = start_prob.shape
    probs = start_prob.reshape(-1,max_len,1) + end_prob.reshape(-1,1,max_len) # array of shape: (batch_size, max_len, max_len), matrix of pairwise sums per each element of the batch
    if min_start is not None:
        mask = np.zeros(probs.shape)  # create a mask to avoid including cases where i > j or i > min_start or j > min_start
        for i,s in enumerate(min_start):
            mask[i,:s,:] = 1
            mask[i,:,:s] = 1
            mask[i][np.tril_indices(max_len,-1)] = 1
        mask[:,0,0] = 0               # we however leave i=j=0 to detect questions without answers
        probs = np.ma.array(probs,mask=mask)
        probs = np.ma.filled(probs,neg_inf)
    else:
        probs = np.triu(probs)
    max_probs = np.argmax(probs.reshape(batch_size,-1), axis=-1) # array of shape: (batch_size,), argmaxes of flattened matrices of pairwise sums
    start_pred, end_pred = np.unravel_index(max_probs, (max_len, max_len)) # two arrays of shape: (batch_size,), 'unflattenning' of max_probs
    return start_pred, end_pred

def convert_predictions_byend(start_prob, end_prob, min_start=None):
    '''
    Return numpy arrays of predictions of indices of starts and ends
    as argmax of unrromalized probabilities over all i>min_start for end and
    as argmax of unrromalized probabilities over all min_start<j<end_pred for start
    '''
    neg_inf = -100
    batch_size, max_len = start_prob.shape
    # first we deal with ends
    if min_start is not None:
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

def get_stats_on_batch(model, batch, with_min_start = True, metrics = ['plain','bysum','byend']):
    start_prob, end_prob = predict(model, batch)
    input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = numpify(*batch)
    batch_size = input_ids.shape[0]
    if with_min_start: min_start = np.argmax(token_type_ids, axis=1)
    else: min_start = None

    metric2convert = {
        'plain' : convert_predictions_plain, 
        'bysum' : convert_predictions_bysum,
        'byend' : convert_predictions_byend
        }

    # batch info
    d = {}
    d['num_examples'] = batch_size
    d['actual_start'] = answer_starts
    d['actual_end'] = answer_ends
    d['actual_label'] = label
    d['input_ids'] = input_ids
    d['indexing'] = indexing
    d['start_probs'] = start_prob
    d['end_probs'] = end_prob

    for metric in metrics:
        start_pred, end_pred = metric2conver[metric](start_prob, end_prob)
        label_pred = np.zeros(batch_size)
        label_pred[start_pred!=0] = 1 
        d[f'guessed_starts_{metric}'] = np.sum(answer_starts == start_pred)
        d[f'guessed_ends_{metric}'] = np.sum(answer_ends == end_pred)
        d[f'guessed_labels_{metric}'] = np.sum(label == label_pred)
        d[f'exact_matches_{metric}'] = np.sum((answer_starts == start_pred) & (answer_ends == end_pred))        
        d[f'predicted_start_{metric}'] = start_pred
        d[f'predicted_end_{metric}'] = end_pred
        d[f'predicted_label_{metric}'] = label_pred

    return d

def evaluate(model, data_mode = 'val', with_min_start = True, metrics = ['plain','bysum','byend'], verbose = True):
    # choose dataloader
    if data_mode == 'train':
        data = model.train_dataloader()
    elif data_mode == 'val':
        data = model.val_dataloader()

    results = {'num_examples' : 0}
    for metric in metrics:
        results[f'guessed_starts_{metric}'] = 0
        results[f'guessed_ends_{metric}'] = 0
        results[f'exact_matches_{metric}'] = 0
        results[f'guessed_labels_{metric}'] = 0

    # iterate over batches
    for batch_id, batch in tqdm(list(enumerate(data))):
        batch_stats = get_stats_on_batch(model, batch, with_min_start = with_min_start, metrics = metrics)
        for key in results.keys():
            results[key] += batch_stats[key]
    
    accuracy = {'num_examples' : 0}
    for metric in metrics:
        accuracy[f'EM_acc_{metric}'] = results[f'exact_matches_{metric}'] / results['num_examples']
        accuracy[f'start_acc_{metric}'] = results[f'guessed_starts_{metric}'] / results['num_examples']
        accuracy[f'end_acc_{metric}'] = results[f'guessed_ends_{metric}'] / results['num_examples']
        accuracy[f'label_acc_{metric}'] = results[f'guessed_labels_{metric}'] / results['num_examples']
    
    if verbose:
        print(f'Evaluation results for with_min_start={with_min_start}:')
        pprint(accuracy)
    return results, accuracy
