import json
import time
import torch
import paddle
import random
import numpy as np
from itertools import chain
from reprod_log import ReprodDiffHelper, ReprodLogger
from tokenizer import LongformerTokenizer as PDTokenizer
from transformers.models.longformer.tokenization_longformer import LongformerTokenizer as PTTokenizer

data_path = "../../data/wikihop/dev.json"
pt_path = "/root/autodl-tmp/models/longformer-base-4096"
pd_path = "/root/autodl-tmp/models/paddle-longformer-base"
seed = 42
max_seq_len = 4096
truncate_seq_len = 1000000000


def torch_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_string(s):
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' !', '!')
    s = s.replace(' ?', '?')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    s = s.replace(" 's", "'s")
    return ' '.join(s.strip().split())


def get_tokenizer(path, phase="paddle"):
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]']
    
    if phase == "paddle":
        tokenizer = PDTokenizer.from_pretrained(path)
    else:
        tokenizer = PTTokenizer.from_pretrained(path)
    tokenizer.add_tokens(additional_tokens)
    return tokenizer


def preprocess_data(path, phase="paddle"):
    print("-----phase={}-----".format(phase))
    tokenizer = get_tokenizer(path, phase)
    
    def tok(s):
        return tokenizer.tokenize(normalize_string(s), add_prefix_space=True)
    
    the_tok = tok
    doc_start = '</s>'
    doc_end = '</s>'
    
    with open(data_path, 'r') as fin:
        data = json.load(fin)
    print("Read data, {} instances".format(len(data)))
    
    t1 = time.time()
    for instance_num, instance in enumerate(data):
        if instance_num % 1000 == 0:
            print("Finished {} instances of {}, total time={}s".format(instance_num, len(data), time.time() - t1))
        query_tokens = ['[question]'] + the_tok(instance['query']) + ['[/question]']
        supports_tokens = [
            [doc_start] + the_tok(support) + [doc_end]
            for support in instance['supports']
        ]
        candidate_tokens = [
            ['[ent]'] + the_tok(candidate) + ['[/ent]']
            for candidate in instance['candidates']
        ]
        answer_index = instance['candidates'].index(instance['answer'])
        
        instance['query_tokens'] = query_tokens
        instance['supports_tokens'] = supports_tokens
        instance['candidate_tokens'] = candidate_tokens
        instance['answer_index'] = answer_index
    
    print("Finished tokenizing")
    with open("dev.tokenized." + phase + ".json", 'w') as fout:
        fout.write(json.dumps(data))


class Torch_WikihopQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, shuffle_candidates):
        super(Torch_WikihopQA_Dataset, self).__init__()
        print("Reading data from {}".format("dev.tokenized.torch.json"))
        with open("dev.tokenized.torch.json", 'r') as fin:
            self.instances = json.load(fin)
        
        self.shuffle_candidates = shuffle_candidates
        self._tokenizer = get_tokenizer(pt_path, phase="torch")
    
    @staticmethod
    def collate_single_item(x):
        # for batch size = 1
        assert len(x) == 1
        return [x[0][0].unsqueeze(0), x[0][1].unsqueeze(0), x[0][2], x[0][3]]
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self._convert_to_tensors(self.instances[idx])
    
    def _convert_to_tensors(self, instance):
        # list of wordpiece tokenized candidates surrounded by [ent] and [/ent]
        candidate_tokens = instance['candidate_tokens']
        # list of word piece tokenized support documents surrounded by </s> </s>
        supports_tokens = instance['supports_tokens']
        query_tokens = instance['query_tokens']
        answer_index = instance['answer_index']
        
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        
        # concat all the candidate_tokens with <s>: <s> + candidates
        all_candidate_tokens = ['<s>'] + query_tokens
        
        # candidates
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
            new_answer_index = sort_order.index(answer_index)
            answer_index = new_answer_index
        all_candidate_tokens.extend(chain.from_iterable([candidate_tokens[k] for k in sort_order]))
        
        # the supports
        n_supports = len(supports_tokens)
        sort_order = list(range(n_supports))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
        all_support_tokens = list(chain.from_iterable([supports_tokens[k] for k in sort_order]))
        
        # convert to ids
        candidate_ids = self._tokenizer.convert_tokens_to_ids(all_candidate_tokens)
        support_ids = self._tokenizer.convert_tokens_to_ids(all_support_tokens)
        
        # get the location of the predicted indices
        predicted_indices = [k for k, token in enumerate(all_candidate_tokens) if token == '[ent]']
        
        # candidate_ids, support_ids, prediction_indices, correct_prediction_index
        """
        return torch.tensor(candidate_ids), \
               torch.tensor(support_ids), \
               torch.tensor(predicted_indices), \
               torch.tensor([answer_index])
        """
        return {
            "candidate_ids": torch.tensor(candidate_ids),
            "support_ids": torch.tensor(support_ids),
            "predicted_indices": torch.tensor(predicted_indices),
            "answer_index": torch.tensor([answer_index]),
        }


class Paddle_WikihopQA_Dataset(paddle.io.Dataset):
    def __init__(self, shuffle_candidates):
        super(Paddle_WikihopQA_Dataset, self).__init__()
        print("Reading data from {}".format("dev.tokenized.paddle.json"))
        with open("dev.tokenized.paddle.json", 'r') as fin:
            self.instances = json.load(fin)
        
        self.shuffle_candidates = shuffle_candidates
        self._tokenizer = get_tokenizer(pd_path, phase="paddle")
    
    @staticmethod
    def collate_single_item(x):
        # for batch size = 1
        assert len(x) == 1
        return [x[0][0].unsqueeze(0), x[0][1].unsqueeze(0), x[0][2], x[0][3]]
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self._convert_to_tensors(self.instances[idx])
    
    def _convert_to_tensors(self, instance):
        # list of wordpiece tokenized candidates surrounded by [ent] and [/ent]
        candidate_tokens = instance['candidate_tokens']
        # list of word piece tokenized support documents surrounded by </s> </s>
        supports_tokens = instance['supports_tokens']
        query_tokens = instance['query_tokens']
        answer_index = instance['answer_index']
        
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        
        # concat all the candidate_tokens with <s>: <s> + candidates
        all_candidate_tokens = ['<s>'] + query_tokens
        
        # candidates
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
            new_answer_index = sort_order.index(answer_index)
            answer_index = new_answer_index
        all_candidate_tokens.extend(chain.from_iterable([candidate_tokens[k] for k in sort_order]))
        
        # the supports
        n_supports = len(supports_tokens)
        sort_order = list(range(n_supports))
        if self.shuffle_candidates:
            random.shuffle(sort_order)
        all_support_tokens = list(chain.from_iterable([supports_tokens[k] for k in sort_order]))
        
        # convert to ids
        candidate_ids = self._tokenizer.convert_tokens_to_ids(all_candidate_tokens)
        support_ids = self._tokenizer.convert_tokens_to_ids(all_support_tokens)
        
        # get the location of the predicted indices
        predicted_indices = [k for k, token in enumerate(all_candidate_tokens) if token == '[ent]']
        
        # candidate_ids, support_ids, prediction_indices, correct_prediction_index
        return paddle.to_tensor(candidate_ids, dtype=paddle.int64), \
               paddle.to_tensor(support_ids, dtype=paddle.int64), \
               paddle.to_tensor(predicted_indices, dtype=paddle.int64), \
               paddle.to_tensor([answer_index], dtype=paddle.int64)


def compare_iter(torch_iter, paddle_iter):
    diff_helper = ReprodDiffHelper()
    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()
    
    for idx, (paddle_batch, torch_batch) in enumerate(zip(paddle_iter, torch_iter)):
        if idx >= 5:
            break
        for i, k in enumerate([0, 1, 2, 3]):  # "candidate_ids", "support_ids", "predicted_indices", "answer_index"
            logger_paddle_data.add(f"dataloader_{idx}_{k}",
                                   paddle_batch[i].squeeze(0).numpy())
            logger_torch_data.add(f"dataloader_{idx}_{k}",
                                  torch_batch[k].cpu().squeeze(0).numpy())
    
    diff_helper.compare_info(logger_paddle_data.data, logger_torch_data.data)
    diff_helper.report(path="wikihop_diff.log")


if __name__ == "__main__":
    torch_set_seed(seed)
    paddle.seed(seed)
    paddle.set_device("cpu")
    # preprocess_data(pt_path, phase="torch")
    # preprocess_data(pd_path, phase="paddle")
    
    torch_dataset = Torch_WikihopQA_Dataset(shuffle_candidates=False)
    paddle_dataset = Paddle_WikihopQA_Dataset(shuffle_candidates=False)
    
    torch_iter = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                             collate_fn=Torch_WikihopQA_Dataset.collate_single_item)
    paddle_iter = paddle.io.DataLoader(paddle_dataset, batch_size=1,
                                       collate_fn=Paddle_WikihopQA_Dataset.collate_single_item)
    
    compare_iter(torch_iter, paddle_iter)
