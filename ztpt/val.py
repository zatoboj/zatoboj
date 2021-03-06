import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm_notebook as tqdm
from .utils import numpify

class Evaluator:
    def __init__(self, model):
        self.model = model
        self.config = model.hparams.config
        self.metrics = model.val_metrics

    def get_stats_on_batch(self, batch):
        input_ids, attention_mask, token_type_ids, label, answer_mask, indexing, answer_starts, answer_ends = numpify(*batch)
        batch_size = input_ids.shape[0]
        min_start = np.argmax(token_type_ids, axis=1)

        # batch statistics
        stats = {
            'num_examples' : batch_size,
            'actual_start' : answer_starts,
            'actual_end' : answer_ends,
            'actual_label' : label,
            'input_ids' : input_ids,
            'indexing' : indexing
        }

        predictions = self.model.get_predictions(batch)
        for metric in self.metrics:
            start_pred, end_pred = self.model.convert_predictions(numpify(*predictions), min_start, metric)
            label_pred = np.zeros(batch_size)
            label_pred[start_pred!=0] = 1 
            stats[f'guessed_starts_{metric}'] = np.sum(answer_starts == start_pred)
            stats[f'guessed_ends_{metric}'] = np.sum(answer_ends == end_pred)
            stats[f'guessed_labels_{metric}'] = np.sum(label == label_pred)
            stats[f'exact_matches_{metric}'] = np.sum((answer_starts == start_pred) & (answer_ends == end_pred))        
            stats[f'predicted_start_{metric}'] = start_pred
            stats[f'predicted_end_{metric}'] = end_pred
            stats[f'predicted_label_{metric}'] = label_pred
            stats[f'contains_answer_{metric}'] = np.sum((answer_starts >= start_pred) & (answer_ends <= end_pred))        
        stats['loss'] = (self.model.compute_loss(predictions, batch)).detach().cpu().numpy()
        return stats

    def evaluate_on_batch(self, batch):
        results = {'num_examples' : 0}
        for metric in self.metrics:
            results[f'guessed_starts_{metric}'] = 0
            results[f'guessed_ends_{metric}'] = 0
            results[f'exact_matches_{metric}'] = 0
            results[f'guessed_labels_{metric}'] = 0
            results[f'contains_answer_{metric}'] = 0

        batch_stats = self.get_stats_on_batch(batch)
        for key in results.keys():
            results[key] += batch_stats[key]

        val_dict = {'val_loss' : batch_stats['loss']}
        for metric in self.metrics:
            val_dict[f'EM_acc/{metric}'] = results[f'exact_matches_{metric}'] / results['num_examples']
            val_dict[f'start_acc/{metric}'] = results[f'guessed_starts_{metric}'] / results['num_examples']
            val_dict[f'end_acc/{metric}'] = results[f'guessed_ends_{metric}'] / results['num_examples']
            val_dict[f'label_acc/{metric}'] = results[f'guessed_labels_{metric}'] / results['num_examples']
            val_dict[f'contains_answer_acc/{metric}'] = results[f'contains_answer_{metric}'] / results['num_examples']
            
        return results, val_dict
        
def evaluate(model, data_mode = 'val', verbose = True):
    evaluator = Evaluator(model)
    # choose dataloader
    if data_mode == 'train':
        data = model.train_dataloader()
    elif data_mode == 'val':
        data = model.val_dataloader()
    # initalize results dictionary
    results = {'num_examples' : 0}

    metrics = model.val_metrics
    for metric in metrics:
        results[f'guessed_starts_{metric}'] = 0
        results[f'guessed_ends_{metric}'] = 0
        results[f'exact_matches_{metric}'] = 0
        results[f'guessed_labels_{metric}'] = 0
    # iterate over batches and update results dictionary
    for batch_id, batch in tqdm(list(enumerate(data))):
        batch_stats = evaluator.get_stats_on_batch(batch)
        for key in results.keys():
            results[key] += batch_stats[key]
    # create accuracy dictionary
    accuracy = {'num_examples' : 0}
    for metric in metrics:
        accuracy[f'EM_acc/{metric}'] = results[f'exact_matches_{metric}'] / results['num_examples']
        accuracy[f'start_acc/{metric}'] = results[f'guessed_starts_{metric}'] / results['num_examples']
        accuracy[f'end_acc/{metric}'] = results[f'guessed_ends_{metric}'] / results['num_examples']
        accuracy[f'label_acc/{metric}'] = results[f'guessed_labels_{metric}'] / results['num_examples']
        
    # print accuracy results
    if verbose:
        print(f'Evaluation results for with_min_start={with_min_start}:')
        pprint(accuracy)        
    return results, accuracy

# TODO: write a function that would print wrong examples
# with coloring of answers and wrong answers
