import os
import json
import time
import torch
import string
import paddle
import random
import numpy as np
from itertools import chain
from reprod_log import ReprodDiffHelper, ReprodLogger
from tokenizer import LongformerTokenizer as PDTokenizer
from transformers.models.longformer.tokenization_longformer import LongformerTokenizer as PTTokenizer

data_path = "/root/autodl-tmp/data/triviaqa/squad-wikipedia-train-4096.json"
pt_path = "/root/autodl-tmp/models/longformer-base-4096"
pd_path = "/root/autodl-tmp/models/paddle-longformer-base"
seed = 42
max_seq_len = 4096
truncate_seq_len = 1000000000
max_doc_len = 4096
max_num_answers = 64
max_question_len = 55
doc_stride = -1
max_answer_length = 30
ignore_seq_with_no_answers = False


class Torch_TriviaQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len):
        super(Torch_TriviaQA_Dataset, self).__init__()
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)['data']
            print(f'done reading file: {self.file_path}')
        self.tokenizer = tokenizer
        
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        
        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in self.file_path:  # only for the evaluation set
            self.val_qid_string_to_int_map = \
                {
                    self._get_qid(entry["paragraphs"][0]['qas'][0]['id']): index
                    for index, entry in enumerate(self.data_json)
                }
        else:
            self.val_qid_string_to_int_map = None
    
    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])
    
    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self, idx):
        entry = self.data_json[idx]
        tensors_list = self.one_example_to_tensors(entry, idx)
        assert len(tensors_list) == 1
        return tensors_list[0]
    
    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        
        tensors_list = []
        for paragraph in example["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            
            for qa in paragraph["qas"]:
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answer_spans = []
                for answer in qa["answers"]:
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    try:
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    except:
                        print(f'Reading example {idx} failed')
                        start_position = 0
                        end_position = 0
                    answer_spans.append({'start': start_position, 'end': end_position})
                
                # ===== Given an example, convert it into tensors  =============
                query_tokens = self.tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:self.max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    # hack: the line below should have been `self.tokenizer.tokenize(token')`
                    # but roberta tokenizer uses a different subword if the token is the beginning of the string
                    # or in the middle. So for all tokens other than the first, simulate that it is not the first
                    # token by prepending a period before tokenizing, then dropping the period afterwards
                    sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)
                
                all_doc_tokens = all_doc_tokens[:self.max_doc_len]
                
                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0
                if self.doc_stride < 0:
                    # negative doc_stride indicates no sliding window, but using first slice
                    self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                start_positions_list = []
                end_positions_list = []
                for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
                    
                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                             + doc_slice_tokens + [self.tokenizer.sep_token]
                    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
                    assert len(segment_ids) == len(tokens)
                    
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    
                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)
                        
                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len
                    
                    doc_offset = len(query_tokens) + 2 - slice_start
                    start_positions = []
                    end_positions = []
                    for answer_span in answer_spans:
                        start_position = answer_span['start']
                        end_position = answer_span['end']
                        tok_start_position_in_doc = orig_to_tok_index[start_position]
                        not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                        tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                        if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                            # this answer is outside the current slice
                            continue
                        start_positions.append(tok_start_position_in_doc + doc_offset)
                        end_positions.append(tok_end_position_in_doc + doc_offset)
                    assert len(start_positions) == len(end_positions)
                    if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                        continue
                    
                    # answers from start_positions and end_positions if > self.max_num_answers
                    start_positions = start_positions[:self.max_num_answers]
                    end_positions = end_positions[:self.max_num_answers]
                    
                    # -1 padding up to self.max_num_answers
                    padding_len = self.max_num_answers - len(start_positions)
                    start_positions.extend([-1] * padding_len)
                    end_positions.extend([-1] * padding_len)
                    
                    # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                    found_start_positions = set()
                    found_end_positions = set()
                    for i, (start_position, end_position) in enumerate(zip(start_positions, end_positions)):
                        if start_position in found_start_positions:
                            start_positions[i] = -1
                        if end_position in found_end_positions:
                            end_positions[i] = -1
                        found_start_positions.add(start_position)
                        found_end_positions.add(end_position)
                    
                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                
                tensors_list.append((torch.tensor(input_ids_list), torch.tensor(input_mask_list),
                                     torch.tensor(segment_ids_list),
                                     torch.tensor(start_positions_list), torch.tensor(end_positions_list),
                                     self._get_qid(qa['id']), qa["aliases"]))  # for eval
        return tensors_list
    
    def _get_qid(self, qid):
        """all input qids are formatted uniqueID__evidenceFile, but for wikipedia, qid = uniqueID,
        and for web, qid = uniqueID__evidenceFile. This function takes care of this conversion.
        """
        if 'wikipedia' in self.file_path:
            # for evaluation on wikipedia, every question has one answer even if multiple evidence documents are given
            return qid.split('--')[0]
        elif 'web' in self.file_path:
            # for evaluation on web, every question/document pair have an answer
            return qid
        elif 'sample' in self.file_path:
            return qid
        else:
            raise RuntimeError('Unexpected filename')
    
    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 2  # qids and aliases
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
        
        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


class Paddle_TriviaQA_Dataset(paddle.io.Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len):
        super(Paddle_TriviaQA_Dataset, self).__init__()
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)['data']
            print(f'done reading file: {self.file_path}')
        self.tokenizer = tokenizer
        
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        
        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in self.file_path:  # only for the evaluation set
            self.val_qid_string_to_int_map = \
                {
                    self._get_qid(entry["paragraphs"][0]['qas'][0]['id']): index
                    for index, entry in enumerate(self.data_json)
                }
        else:
            self.val_qid_string_to_int_map = None
    
    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])
    
    def __len__(self):
        return len(self.data_json)
    
    def __getitem__(self, idx):
        entry = self.data_json[idx]
        tensors_list = self.one_example_to_tensors(entry, idx)
        assert len(tensors_list) == 1
        return tensors_list[0]
    
    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        
        tensors_list = []
        for paragraph in example["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            
            for qa in paragraph["qas"]:
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answer_spans = []
                for answer in qa["answers"]:
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    try:
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    except:
                        print(f'Reading example {idx} failed')
                        start_position = 0
                        end_position = 0
                    answer_spans.append({'start': start_position, 'end': end_position})
                
                # ===== Given an example, convert it into tensors  =============
                query_tokens = self.tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:self.max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    # hack: the line below should have been `self.tokenizer.tokenize(token')`
                    # but roberta tokenizer uses a different subword if the token is the beginning of the string
                    # or in the middle. So for all tokens other than the first, simulate that it is not the first
                    # token by prepending a period before tokenizing, then dropping the period afterwards
                    sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)
                
                all_doc_tokens = all_doc_tokens[:self.max_doc_len]
                
                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0
                if self.doc_stride < 0:
                    # negative doc_stride indicates no sliding window, but using first slice
                    self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                start_positions_list = []
                end_positions_list = []
                for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
                    
                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                             + doc_slice_tokens + [self.tokenizer.sep_token]
                    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
                    assert len(segment_ids) == len(tokens)
                    
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    
                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)
                        
                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len
                    
                    doc_offset = len(query_tokens) + 2 - slice_start
                    start_positions = []
                    end_positions = []
                    for answer_span in answer_spans:
                        start_position = answer_span['start']
                        end_position = answer_span['end']
                        tok_start_position_in_doc = orig_to_tok_index[start_position]
                        not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                        tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                        if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                            # this answer is outside the current slice
                            continue
                        start_positions.append(tok_start_position_in_doc + doc_offset)
                        end_positions.append(tok_end_position_in_doc + doc_offset)
                    assert len(start_positions) == len(end_positions)
                    if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                        continue
                    
                    # answers from start_positions and end_positions if > self.max_num_answers
                    start_positions = start_positions[:self.max_num_answers]
                    end_positions = end_positions[:self.max_num_answers]
                    
                    # -1 padding up to self.max_num_answers
                    padding_len = self.max_num_answers - len(start_positions)
                    start_positions.extend([-1] * padding_len)
                    end_positions.extend([-1] * padding_len)
                    
                    # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                    found_start_positions = set()
                    found_end_positions = set()
                    for i, (start_position, end_position) in enumerate(zip(start_positions, end_positions)):
                        if start_position in found_start_positions:
                            start_positions[i] = -1
                        if end_position in found_end_positions:
                            end_positions[i] = -1
                        found_start_positions.add(start_position)
                        found_end_positions.add(end_position)
                    
                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                
                tensors_list.append((
                    paddle.to_tensor(input_ids_list, dtype=paddle.int64),
                    paddle.to_tensor(input_mask_list, dtype=paddle.int64),
                    paddle.to_tensor(segment_ids_list, dtype=paddle.int64),
                    paddle.to_tensor(start_positions_list, dtype=paddle.int64),
                    paddle.to_tensor(end_positions_list, dtype=paddle.int64),
                    self._get_qid(qa['id']), qa["aliases"]
                ))  # for eval
        return tensors_list
    
    def _get_qid(self, qid):
        """all input qids are formatted uniqueID__evidenceFile, but for wikipedia, qid = uniqueID,
        and for web, qid = uniqueID__evidenceFile. This function takes care of this conversion.
        """
        if 'wikipedia' in self.file_path:
            # for evaluation on wikipedia, every question has one answer even if multiple evidence documents are given
            return qid.split('--')[0]
        elif 'web' in self.file_path:
            # for evaluation on web, every question/document pair have an answer
            return qid
        elif 'sample' in self.file_path:
            return qid
        else:
            raise RuntimeError('Unexpected filename')
    
    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 2  # qids and aliases
        fields = [x for x in zip(*batch)]
        stacked_fields = [paddle.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors
        
        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


def get_dataset():
    pt_tokenizer = PTTokenizer.from_pretrained(pt_path)
    pd_tokenizer = PDTokenizer.from_pretrained(pd_path)
    
    torch_dataset = Torch_TriviaQA_Dataset(file_path=data_path, tokenizer=pt_tokenizer,
                                           max_seq_len=max_seq_len, max_doc_len=max_doc_len,
                                           doc_stride=doc_stride,
                                           max_num_answers=max_num_answers,
                                           max_question_len=max_question_len,
                                           ignore_seq_with_no_answers=ignore_seq_with_no_answers)
    
    paddle_dataset = Paddle_TriviaQA_Dataset(file_path=data_path, tokenizer=pd_tokenizer,
                                             max_seq_len=max_seq_len, max_doc_len=max_doc_len,
                                             doc_stride=doc_stride,
                                             max_num_answers=max_num_answers,
                                             max_question_len=max_question_len,
                                             ignore_seq_with_no_answers=ignore_seq_with_no_answers)
    return torch_dataset, paddle_dataset


def compare_iter(torch_dataset, paddle_dataset):
    diff_helper = ReprodDiffHelper()
    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()
    
    torch_iter = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Torch_TriviaQA_Dataset.collate_one_doc_and_lists)
    
    paddle_iter = paddle.io.DataLoader(
        paddle_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Paddle_TriviaQA_Dataset.collate_one_doc_and_lists)
    
    for idx, (paddle_batch, torch_batch) in enumerate(zip(paddle_iter, torch_iter)):
        if idx >= 5:
            break
        for i, k in enumerate([0, 1, 2, 3, 4]):  # "candidate_ids", "support_ids", "predicted_indices", "answer_index"
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[i].squeeze(0).numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k].cpu().squeeze(0).numpy())
    
    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report(path="triviaqa_diff.log")


if __name__ == "__main__":
    paddle.set_device("cpu")
    torch_dataset, paddle_dataset = get_dataset()
    compare_iter(torch_dataset, paddle_dataset)
