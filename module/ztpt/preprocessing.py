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
                    answer_start = qa["plausible_answers"][0]["answer_start"]
                    answer_text = qa["plausible_answers"][0]["text"]
                answer_end = answer_start + len(answer_text)
                preprocessed_data.append((question_text, paragraph_text, label, answer_start, answer_end, answer_text))
    preprocessed_data = shuffle(preprocessed_data, random_state=420)
    return preprocessed_data

def tokenize_data(preprocessed_data: list, conf: dict) -> dict:
    '''
    Tokenize data that was organized in a list ot tuples using tokenizer coming from configuration. Return data as a dictionary (see the dictionary structure at the end of this function).
    '''
    input_ids, token_type_ids, labels = [], [], [] 
    attention_mask, answer_mask, plausible_answer_mask = [], [], []
    full_questions, actual_answers, full_paragraphs, plausible_answers = [], [], [], []
    answer_starts, answer_ends = [], []
    max_len = conf['max_len']
    tokenizer = get_tokenizer(conf)

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

def preprocess(conf):
    '''
    Run all preprocessing steps for 'train-v2.0.json' and 'val-v2.0.json' files if the preprocessing has not already been done for a given configuration.
    '''
    conf_prefix = conf['model'] + '-' + conf['model_version'] + '-' + str(conf['max_len'])   
    base_dir_read = conf['base_dir_read']
    base_dir_write = conf['base_dir_write']
    # first deal with train-test data
    train_input_path = base_dir_read + 'train-v2.0.json'
    train_data_path = base_dir_write + conf_prefix + '-train.pickle'
    test_data_path = base_dir_write + conf_prefix + '-test.pickle'
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print(f'Preprocessed train and test data files already exist as {train_data_path} and {test_data_path}.')
    else:               
        if os.path.exists(train_input_path):
            with open(train_input_path, "r") as reader:
                input_data = json.load(reader)["data"]
        else:
            raise ValueError(f'Input train data file {train_input_path} does not exist. Please, upload the file to the folder.')   
        data_train, data_test = train_test_split(input_data, test_size=0.05, random_state=420)
        data_train, data_test = organize_raw_data(data_train), organize_raw_data(data_test)
        data_train, data_test = tokenize_data(data_train, conf), tokenize_data(data_test, conf)
        pickle.dump(data_train, train_data_path, 'wb')
        pickle.dump(data_test, test_data_path, 'wb')
        print(f'Preprocessed data succesfully saved as {train_data_name} and {test_data_name}.')
    # now deal with val data    
    val_input_path = base_dir_read + 'val-v2.0.json'
    val_data_path = base_dir_write + conf_prefix + '-val.pickle'
    if os.path.exists(train_data_path):
        print(f'Preprocessed validation data file already exists as {val_data_path}.')
    else:
        if os.path.exists(val_input_path):
            with open(val_input_path, "r") as reader:
                input_data = json.load(reader)["data"]
        else:
            raise ValueError(f'Input validation data file {val_input_path} does not exist. Please, upload the file to the folder.')
        data_val = organize_raw_data(data_val)
        data_val = tokenize_data(data_val, conf)
        pickle.dump(data_val, val_data_path, 'wb')
        print(f'Preprocessed validation data succesfully saved as {val_data_path}.')