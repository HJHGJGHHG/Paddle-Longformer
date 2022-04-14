# STEP0-权重转换、Tokenizer 对齐
## 1. 权重转换
&emsp;&emsp;使用权重转换脚本  
```
python torch2paddle.py --torch_file="longformer-base/pytorch_model.bin" --paddle_file="longformer-base-paddle/model_state.pdparams"

python torch2paddle.py --torch_file "longformer-large/pytorch_model.bin" --paddle_file "longformer-large-paddle/model_state.pdparams"
```

&emsp;&emsp;也可以用我传到百度网盘上的模型：

## 2. Tokenizer 对齐
&emsp;&emsp;Longformer 的 Tokenizer 是基于 BPE 的，直接继承 RobertaBPETokenizer 就好，测试脚本：  
```
python tokenizer.py
```
&emsp;&emsp;结果：  
```
{'input_ids': [0, 243, 16, 10, 2579, 183, 452, 2156, 38, 236, 7, 213, 7, 5, 2221, 27785, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [0, 243, 16, 10, 2579, 183, 452, 2156, 38, 236, 7, 213, 7, 5, 2221, 27785, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
{'input_ids': [0, 243, 16, 10, 2579, 183, 452, 2156, 38, 236, 7, 213, 7, 5, 2221, 27785, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [0, 243, 16, 10, 2579, 183, 452, 2156, 38, 236, 7, 213, 7, 5, 2221, 27785, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

&emsp;&emsp;如 2、4 行所示，复现的 LongformerTokenizer 是默认不得到 attention mask 的，如需使用，请添加参数 return_attention_mask=True：  
```
encoding = tokenizer.encode(question, text, return_attention_mask=True)
```