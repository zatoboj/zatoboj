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

def convert_predictions_dumb(start_prob, end_prob):
    '''
    Return numpy arrays of predictions of indices of starts and ends
    as argmax of unnormalized probability vectors.
    '''
    start_pred = np.argmax(start_prob, axis=1)
    end_pred = np.argmax(end_prob, axis=1)
    return start_pred, end_pred

def convert_predictions_smart(start_prob, end_prob, min_start=None):
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

def get_stats_on_batch(model, batch, with_min_start = True):
    start_prob, end_prob = predict(model, batch)
    input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = numpify(*batch)
    batch_size = input_ids.shape[0]
    if with_min_start: 
        min_start = np.argmax(token_type_ids, axis=1)
    else:
        min_start = None
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

    # convert mode 'dumb'
    start_pred, end_pred = convert_predictions_dumb(start_prob, end_prob)
    label_pred = np.zeros(batch_size)
    label_pred[start_pred!=0] = 1 
    d['guessed_starts_dumb'] = np.sum(answer_starts == start_pred)
    d['guessed_ends_dumb'] = np.sum(answer_ends == end_pred)
    d['exact_matches_dumb'] = np.sum((answer_starts == start_pred) & (answer_ends == end_pred))
    d['predicted_label_dumb'] = start_pred
    d['predicted_label_dumb'] = end_pred
    d['predicted_label_dumb'] = label_pred
    d['guessed_labels_dumb'] = np.sum(label == label_pred)

    # convert mode 'smart'
    start_pred, end_pred = convert_predictions_smart(start_prob, end_prob, min_start=min_start)
    label_pred = np.zeros(batch_size)
    label_pred[start_pred!=0] = 1 
    d['guessed_starts_smart'] = np.sum(answer_starts == start_pred)
    d['guessed_ends_smart'] = np.sum(answer_ends == end_pred)
    d['exact_matches_smart'] = np.sum((answer_starts == start_pred) & (answer_ends == end_pred))
    d['predicted_label_smart'] = start_pred
    d['predicted_label_smart'] = end_pred
    d['predicted_label_smart'] = label_pred
    d['guessed_labels_smart'] = np.sum(label == label_pred)

    return d

def evaluate(model, data_mode='val', verbose=True, with_min_start=True):
    # choose dataloader
    if data_mode == 'train':
        data = model.train_dataloader()
    elif data_mode == 'val':
        data = model.val_dataloader()

    results = {
            'num_examples' : 0,
            'guessed_starts_dumb' : 0,
            'guessed_ends_dumb' : 0,
            'exact_matches_dumb' : 0,
            'guessed_labels_dumb' : 0,
            'guessed_starts_smart' : 0,
            'guessed_ends_smart' : 0,
            'exact_matches_smart' : 0.,
            'guessed_labels_smart' : 0
        }

    accuracy = {}

    # iterate over batches
    for batch_id, batch in tqdm(list(enumerate(data))):
        batch_stats = get_stats_on_batch(model, batch, with_min_start=with_min_start)
        for key in results.keys():
            results[key] += batch_stats[key]
    accuracy['EM_acc_dumb'] = results['exact_matches_dumb'] / results['num_examples']
    accuracy['EM_acc_smart'] = results['exact_matches_smart'] / results['num_examples']
    accuracy['start_acc_dumb'] = results['guessed_starts_dumb'] / results['num_examples']
    accuracy['start_acc_smart'] = results['guessed_starts_smart'] / results['num_examples']
    accuracy['end_acc_dumb'] = results['guessed_ends_dumb'] / results['num_examples']
    accuracy['end_acc_smart'] = results['guessed_ends_smart'] / results['num_examples']
    accuracy['label_acc_dumb'] = results['guessed_labels_dumb'] / results['num_examples']
    accuracy['label_acc_smart'] = results['guessed_labels_smart'] / results['num_examples']
    accuracy['num_examples'] = results['num_examples']
    if verbose:
        print(f'Evaluation results for with_min_start={with_min_start}:')
        pprint(accuracy)
    return results, accuracy

