import pickle, json
import os
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from .utils import get_tokenizer

def organize_raw_data(input_data: list) -> list:
    '''
    Organize raw data obtained from json file into a list ot tuples:
    (question_text, paragraph_text, label, answer_start, answer_end, answer_text).
    '''
    preprocessed_data = list()
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            for qa in paragraph["qas"]:
                question_text = qa['question']
                label = qa["is_impossible"]
                if not label:
                    answer_start = qa["answers"][0]["answer_start"]
                    answer_text = qa["answers"][0]["text"]
                else:
                    plaus_answer = qa["plausible_answers"]
                    if plaus_answer:
                        answer_start = plaus_answer[0]["answer_start"]
                        answer_text = plaus_answer[0]["text"]
                    else:
                        answer_start = 0
                        answer_text = ''
                answer_end = answer_start + len(answer_text)
                preprocessed_data.append((question_text, paragraph_text, label, answer_start, answer_end, answer_text))
    preprocessed_data = shuffle(preprocessed_data, random_state=420)
    return preprocessed_data

def tokenize_data(preprocessed_data: list, config: dict) -> dict:
    '''
    Tokenize data that was organized in a list ot tuples using tokenizer coming from configuration. Return data as a dictionary (see the dictionary structure at the end of this function).
    '''
    input_ids, token_type_ids, labels = [], [], [] 
    attention_mask, answer_mask, plausible_answer_mask = [], [], []
    full_questions, actual_answers, full_paragraphs, plausible_answers = [], [], [], []
    answer_starts, answer_ends = [], []
    max_len = config.model.max_len
    tokenizer = get_tokenizer(config)

    print('Beginning to tokenize data...')
    for step in tqdm_notebook(preprocessed_data):
        question, paragraph, label, answer_start, answer_end, answer_text = step
        # tokenizing the question
        tmp_question = tokenizer.tokenize(question)
        # tokenizing the paragraph. tokenizing the part before the answer, the answer, and after the answer separately to keep track of tokens corresponding to the answer
        tmp_preanswer = tokenizer.tokenize(paragraph[0:answer_start])
        tmp_answer = tokenizer.tokenize(paragraph[answer_start:answer_end])
        tmp_postanswer = tokenizer.tokenize(paragraph[answer_end:])
        tmp_paragraph = tmp_preanswer + tmp_answer + tmp_postanswer

        len_q, len_p = len(tmp_question), len(tmp_paragraph)

        if len_q + len_p + 3 <= max_len:
            input_ids.append(tokenizer.convert_tokens_to_ids(["[CLS]"] + tmp_question + ["[SEP]"] + tmp_paragraph + ["[SEP]"]) + [0]*(max_len - (len_q + len_p + 3)))
            token_type_ids.append([0 for i in range(len_q + 2)] + [1 for i in range(max_len - (len_q + 2))]) 
            attention_mask.append([1 for i in range(len_q + len_p + 3)] + [0 for i in range(len_q + len_p + 3, max_len)])    
            answer_start = len_q + len(tmp_preanswer) + 2 
            answer_end = len_q + len(tmp_preanswer) + len(tmp_answer) + 2
            plausible_answer_mask.append([0 for i in range(answer_start)] + [1 for i in range(answer_start, answer_end)] + [0 for i in range(answer_end, max_len)])
            if label:
                answer_mask.append([0 for i in range(max_len)])
                answer_starts.append(0)
                answer_ends.append(0)
                actual_answers.append('no answer')
                plausible_answers.append(answer_text)
            else:
                answer_mask.append([0 for i in range(answer_start)] + [1 for i in range(answer_start, answer_end)] + [0 for i in range(answer_end, max_len)])
                answer_starts.append(answer_start)
                answer_ends.append(answer_end-1)
                plausible_answers.append('no answer')
                actual_answers.append(answer_text)
            labels.append(label)
            full_questions.append(question)
            full_paragraphs.append(paragraph)
    print('Succesfully finished tokenizing data.')    
  
    dict_data = {
        "input_ids": input_ids, 
        "token_type_ids": token_type_ids, 
        "labels": labels, 
        "attention_mask": attention_mask, 
        'answer_mask' : answer_mask, 
        'plausible_answer_mask' : plausible_answer_mask,
        'actual_answers' : actual_answers,
        'full_questions' : full_questions,
        'full_paragraphs' : full_paragraphs,
        'plausible_answers' : plausible_answers,
        'answer_starts' : answer_starts,
        'answer_ends' : answer_ends
        }
    return dict_data

def preprocess(config):
    '''
    Run all preprocessing steps for 'train-v2.0.json' and 'val-v2.0.json' files if the preprocessing has not already been done for a given configuration.
    '''
    data_prefix = config.transformer.model + '-' + config.transformer.version + '-' + str(config.model.max_len)   
    data_dir = config.dirs.data
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a datasets folder ('datasets'). Be a good boy, copy shared datasets folder into root directory.")

    # first deal with train data
    train_input_path = data_dir + 'train-v2.0.json'
    train_data_path = data_dir + data_prefix + '-train.pickle'
    if os.path.exists(train_data_path):
        print(f'Preprocessed train data already exist at {train_data_path}.')
    else:               
        if os.path.exists(train_input_path):
            with open(train_input_path, "r") as reader:
                train_data = json.load(reader)["data"]
        else:
            raise ValueError(f'Input train data file {train_input_path} does not exist. Please, upload the file to the folder.')   
        train_data = organize_raw_data(train_data)
        train_data = tokenize_data(train_data, config)
        with open(train_data_path, 'wb') as f:
            pickle.dump(train_data, f)
        print(f'Preprocessed train data succesfully saved at {train_data_path}.')
    # now deal with val/test data (20%/80% split of 'dev-v2.0.json' file)
    dev_input_path = data_dir + 'dev-v2.0.json'
    val_data_path = data_dir + data_prefix + '-val.pickle'
    test_data_path = data_dir + data_prefix + '-test.pickle'
    if os.path.exists(val_data_path) and os.path.exists(test_data_path):
        print(f'Preprocessed validation and test data already exists at {val_data_path} and {test_data_path}.')
    else:
        if os.path.exists(dev_input_path):
            with open(dev_input_path, "r") as reader:
                dev_data = json.load(reader)["data"]
        else:
            raise ValueError(f'Input dev data file {dev_input_path} does not exist. Please, upload the file to the folder.')
        val_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=420)
        val_data, test_data = organize_raw_data(val_data), organize_raw_data(test_data)
        val_data, test_data = tokenize_data(val_data, config), tokenize_data(test_data, config)
        with open(val_data_path, 'wb') as f:
            pickle.dump(val_data, f)
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f'Preprocessed validation and test data succesfully saved at {val_data_path} and {test_data_path}.')

def load_data(config):
    preprocess(config)
    print('Beginning to load preprocessed data...')
    data_dir = config.dirs.data
    if not os.path.exists(data_dir):
        raise FileNotFoundError("Your root directory ('ybshmmlchk') is missing a datasets folder ('datasets'). Be a good boy, copy shared datasets folder into root directory.")
    data_prefix = config.transformer.model + '-' + config.transformer.version + '-' + str(config.model.max_len) 
    train_data_path = data_dir + data_prefix + '-train.pickle'
    val_data_path = data_dir + data_prefix + '-val.pickle'
    test_data_path = data_dir + data_prefix + '-test.pickle'
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_data_path, 'rb') as f:
        val_data = pickle.load(f) 
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    train_data['indexing'] = list(range(len(train_data['labels'])))
    val_data['indexing'] = list(range(len(val_data['labels'])))   
    test_data['indexing'] = list(range(len(test_data['labels'])))   
    print('Preprocessed data has been succesfully loaded')
    print('Train data size:'.ljust(21), len(train_data['labels']))
    print('Validation data size:'.ljust(21), len(val_data['labels']))
    print('Test data size:'.ljust(21), len(test_data['labels']))                   
    return train_data, val_data, test_data