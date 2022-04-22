import json
import time
import paddle
import paddle.nn as nn

from typing import List
from itertools import chain
from paddle.io import Dataset, DataLoader

from tokenizer import LongformerTokenizer
from modeling import LongformerPreTrainedModel, LongformerModel


def normalize_string(s):
    s = s.replace(' .', '.')
    s = s.replace(' ,', ',')
    s = s.replace(' !', '!')
    s = s.replace(' ?', '?')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    s = s.replace(" 's", "'s")
    return ' '.join(s.strip().split())


def get_tokenizer(path):
    additional_tokens = ['[question]', '[/question]', '[ent]', '[/ent]']
    tokenizer = LongformerTokenizer.from_pretrained(path)
    tokenizer.add_tokens(additional_tokens)
    return tokenizer


def preprocess_data(args):
    tokenizer = get_tokenizer(args.model_name_or_path)
    
    def tok(s):
        return tokenizer.tokenize(normalize_string(s), add_prefix_space=True)
    
    the_tok = tok
    doc_start = '</s>'
    doc_end = '</s>'
    
    for phase in ["train", "dev"]:
        with open(args.data_path + phase + ".json", 'r') as fin:
            data = json.load(fin)
        print("Processing {}.json file...".format(phase))
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
        with open(phase + ".tokenized.json", 'w') as fout:
            fout.write(json.dumps(data))


class WikihopQA_Dataset(Dataset):
    def __init__(self, args, file_dir):
        super(WikihopQA_Dataset, self).__init__()
        with open(file_dir, 'r') as fin:
            self.instances = json.load(fin)
        
        self.args = args
        self._tokenizer = args.tokenizer
    
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
        
        # concat all the candidate_tokens with <s>: <s> + candidates
        all_candidate_tokens = ['<s>'] + query_tokens
        
        # candidates
        n_candidates = len(candidate_tokens)
        sort_order = list(range(n_candidates))
        all_candidate_tokens.extend(chain.from_iterable([candidate_tokens[k] for k in sort_order]))
        
        # the supports
        n_supports = len(supports_tokens)
        sort_order = list(range(n_supports))
        all_support_tokens = list(chain.from_iterable([supports_tokens[k] for k in sort_order]))
        
        # convert to ids
        candidate_ids = self._tokenizer.convert_tokens_to_ids(all_candidate_tokens)
        support_ids = self._tokenizer.convert_tokens_to_ids(all_support_tokens)
        
        # get the location of the predicted indices
        predicted_indices = [k for k, token in enumerate(all_candidate_tokens) if token == '[ent]']
        
        # candidate_ids, support_ids, predicted_indices, answer_index
        return {
            "candidate_ids": paddle.to_tensor(candidate_ids, dtype=paddle.int64),
            "support_ids": paddle.to_tensor(support_ids, dtype=paddle.int64),
            "predicted_indices": paddle.to_tensor(predicted_indices, dtype=paddle.int64),
            "answer_index": paddle.to_tensor([answer_index], dtype=paddle.int64),
        }


def get_iter(train_dataset, dev_dataset):
    train_iter = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )
    dev_iter = DataLoader(
        dev_dataset,
        batch_size=1,
        shuffle=False
    )
    return train_iter, dev_iter


class WikihopQAModel(LongformerPreTrainedModel):
    def __init__(self, args):
        super(WikihopQAModel, self).__init__()
        self.args = args
        self.longformer = LongformerModel.from_pretrained(args.model_name_or_path, add_pooling_layer=False)
        self.answer_score = nn.Linear(self.longformer.embeddings.word_embeddings.weight.shape[1], 1, bias_attr=False)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
        self._truncate_seq_len = self.args.truncate_seq_len
        if self._truncate_seq_len is None:
            # default is to use all context
            self._truncate_seq_len = 1000000000
        
        # Initialize weights and apply final processing
        self.init_weights()
    
    def forward(
            self,
            candidate_ids,
            support_ids,
            predicted_indices,
            answer_index,
    ):
        activations = self.get_activations(candidate_ids, support_ids, self.args.max_length, self._truncate_seq_len)
        # activations is a list of activations [(batch_size, max_length (or shorter), embed_dim)]
        # select the activations we will make predictions at from each element of the list.
        # we are guaranteed the predicted_indices are valid indices since each element
        # of activations list has all of the candidates
        prediction_activations = [paddle.index_select(act, index=predicted_indices.squeeze(), axis=1) for act in
                                  activations]
        prediction_scores = [
            self.answer_score(prediction_act).squeeze(-1)
            for prediction_act in prediction_activations
        ]
        # prediction_scores is a list of tensors, each is (batch_size, num_predictions)
        # sum across the list for each possible prediction
        sum_prediction_scores = paddle.sum(paddle.concat(
            [pred_scores.unsqueeze(-1) for pred_scores in prediction_scores], axis=-1
        ), axis=-1)
        loss = self.loss_func(sum_prediction_scores, answer_index.squeeze(0))
        
        return loss, sum_prediction_scores
    
    def get_activations(self, candidate_ids, support_ids, max_seq_len, truncate_seq_len) -> List:
        # max_seq_len: the maximum sequence length possible for the model
        # truncate_seq_len: only use the first truncate_seq_len total tokens in the candidate + supports (e.g. just the first 4096)
        candidate_len = candidate_ids.shape[1]
        support_len = support_ids.shape[1]
        
        # attention_mask = 1 for local, 2 for global, 0 for padding (which we can ignore as always batch size=1)
        if candidate_len + support_len <= max_seq_len:
            token_ids = paddle.concat([candidate_ids, support_ids], axis=1)
            attention_mask = paddle.ones_like(token_ids, dtype=paddle.int64)
            
            # get global attention
            global_attention_mask = paddle.zeros_like(attention_mask, dtype=paddle.int64)
            # global attention to all candidates
            global_attention_mask[0, :candidate_len] = 1
            
            return [self.longformer(
                token_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,)[0]]
        else:
            all_activations = []
            available_support_len = max_seq_len - candidate_len
            for start in range(0, support_len, available_support_len):
                end = min(start + available_support_len, support_len, truncate_seq_len)
                token_ids = paddle.concat([candidate_ids, support_ids[:, start:end]], axis=1)
                attention_mask = paddle.ones_like(token_ids, dtype=paddle.int64)
                
                # get global attention
                global_attention_mask = paddle.zeros_like(attention_mask, dtype=paddle.int64)
                # global attention to all candidates
                global_attention_mask[0, :candidate_len] = 1
                
                activations = self.longformer(
                    token_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,)[0]
                all_activations.append(activations)
                if end == truncate_seq_len:
                    break
            
            return all_activations
