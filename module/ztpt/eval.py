import torch
import numpy as np

def numpify(*tensors):
    np_arrays = []
    for tensor in tensors:
        np_arrays.append(tensor.detach().cpu().numpy())
    return np_arrays

# return "probabilities" (in numpy) of start and end. not actually probabilities, because this is before softmax
def predict(model, batch):
    input_ids, attention_mask, token_type_ids, _, _, _, _, _ = batch
    with torch.no_grad():
        start_prob, end_prob = model(input_ids, attention_mask, token_type_ids)
    start_prob, end_prob = numpify(start_prob, end_prob)
    return start_prob, end_prob

# return start and end vectors, just based on argmax taken individually
# here start_prob and end_prob are numpy arrays
def convert_predictions_dumb(start_prob, end_prob):
    start_pred = np.argmax(start_prob, axis=1)
    end_pred = np.argmax(end_prob, axis=1)
    return start_pred, end_pred

# here start_prob and end_prob are numpy arrays
def convert_predictions_smart(start_prob, end_prob, min_start=None):
    if min_start is not None: min_start = numpify(min_start)
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
    starts, ends = np.unravel_index(max_probs, (max_len, max_len)) # two arrays of shape: (batch_size,), 'unflattenning' of max_probs
    return start_pred, end_pred

def get_stats_on_batch(model, batch):
    start_prob, end_prob = predict(model, batch)
    input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = numpify(batch)
    batch_size = input_ids.shape[0]
    min_start = np.argmax(token_type_ids, axis=1)

    # batch info
    d['num_examples'] = batch_size
    d['actual_start'] = answer_starts
    d['actual_end'] = answer_ends
    d['actual_label'] = label
    d['input_ids'] = input_ids
    d['indexing'] = indexing
    d['start_probs'] = start_probs
    d['end_probs'] = end_probs

    # convert mode 'dumb'
    start_pred, end_pred = convert_predictions_dumb(start_prob, end_prob)
    label_pred = np.zeros(batch_size)
    label_pred[start_pred!=0] = 1 
    d['guessed_starts_dumb'] = np.sum(answer_starts == start_pred)
    d['guessed_ends_dumb'] = np.sum(answer_ends == end_pred)
    d['exact_matches_dumb'] = np.sum(((answer_starts == start_pred) & (answer_ends == end_pred))
    d['predicted_label_dumb'] = start_pred
    d['predicted_label_dumb'] = end_pred
    d['predicted_label_dumb'] = label_pred
    d['guessed_labels_dumb'] = np.sum(label == label_pred)

    # convert mode 'smart'
    start_pred, end_pred = convert_predictions_smart(start_prob, end_prob, min_start)
    label_pred = np.zeros(batch_size)
    label_pred[start_pred!=0] = 1 
    d['guessed_starts_smart'] = np.sum(answer_starts == start_pred)
    d['guessed_ends_smart'] = np.sum(answer_ends == end_pred)
    d['exact_matches_smart'] = np.sum(((answer_starts == start_pred) & (answer_ends == end_pred))
    d['predicted_label_smart'] = start_pred
    d['predicted_label_smart'] = end_pred
    d['predicted_label_smart'] = label_pred
    d['guessed_labels_smart'] = np.sum(label == label_pred)

    return d

def add_dicts(d1, d2):
    for key in set(d1.keys()).intersection(set(d2.keys())):
        d1[key]+=d2[key]
    return d1

def validate(model, mode = 'val', verbose = True):
    # choose the right dataloader
    if mode == 'train':
        data = model.train_dataloader()
    elif mode == 'val':
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

    # iterate over batches
    for batch_id, batch in enumerate(data):
        batch_stats = get_stats_on_batch(model, batch)
        for key in results.keys():
            results[key] += batch_stats[key]
        results['EM_acc_dumb'] = d['num_exact_matches_guessed'] / d['num_examples']
        d['EM_acc_smart'] = d['num_exact_matches_guessed_post'] / d['num_examples']
        if verbose and batch_id%250 == 0:
            print(batch_id)
            pprint(d)
    return d

