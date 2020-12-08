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

def get_stats_on_batch(model, batch, convert_mode = 'smart'):
    start_prob, end_prob = predict(model, batch)
    input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = numpify(batch)
    d['batch_size'] = input_ids.shape[0]

    if convert_mode == 'dumb':
        strat_pred, end_pred = convert_predictions_dumb(strat_prob, end_prob)
        d['num_starts_guessed_dumb'] = np.sum(answer_starts == start_pred)
        d['num_ends_guessed_dumb'] = np.sum(answer_ends == end_pred)
        d['num_exact_matches_dumb'] = np.sum(((answer_starts == start_pred) & (answer_ends == end_pred))
    elif convert_mode == 'smart':
        strat_pred, end_pred = convert_predictions_smart(strat_prob, end_prob)
        d['num_starts_guessed_smart'] = np.sum(answer_starts == start_pred)
        d['num_ends_guessed_smart'] = np.sum(answer_ends == end_pred)
        d['num_exact_matches_smart'] = np.sum(((answer_starts == start_pred) & (answer_ends == end_pred))
    
    
    
    min_start = np.argmax(token_type_ids, axis=1)

    predicted_label = predicted_start.copy().astype(int)
    predicted_label[predicted_label!=0] = 1 
    label = npf(label).astype(int)
    # d['num_labels_guessed'] = np.sum(label == predicted_label)
    # d['predicted_start'] = predicted_start
    # d['predicted_end'] = predicted_end
    # d['predicted_label'] = predicted_label
    # d['actual_start'] = answer_starts
    # d['actual_end'] = answer_ends
    # d['actual_label'] = label
    # d['input_ids'] = input_ids
    # d['indexing'] = indexing
    # d['start_probs'] = l1
    # d['end_probs'] = l2
    return d

def add_dicts(d1, d2):
  for key in set(d1.keys()).intersection(set(d2.keys())):
    d1[key]+=d2[key]
  return d1

def validate(model, s = 'val'):
  # choose the right dataloader
  if s == 'train':
    a = (model.train_dataloader())
  else:
    a = (model.val_dataloader())

  d = {
        'num_examples' : 0,
        'num_starts_guessed' : 0,
        'num_ends_guessed' : 0,
        'num_exact_matches_guessed' : 0,
        'num_starts_guessed_post' : 0,
        'num_ends_guessed_post' : 0,
        'num_exact_matches_guessed_post' : 0.,
        'num_labels_guessed' : 0
      }

  # iterate over batches
  for batch_ndx, batch in enumerate(a):
    batch_stats = get_stats_on_batch(model, batch)
    d = add_dicts(d, batch_stats)
    d['EM'] = d['num_exact_matches_guessed'] / d['num_examples']
    d['EM post'] = d['num_exact_matches_guessed_post'] / d['num_examples']
    if batch_ndx%300 == 0:
      print(batch_ndx)
      pprint(d)
  return d

