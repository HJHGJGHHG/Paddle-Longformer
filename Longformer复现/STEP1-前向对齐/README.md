# STEP1-前向对齐
## 1. 引言
&emsp;&emsp;本部分将进行前向对齐，以此验证模型组网正确性。为了简便考虑，使用 Longformer 经典应用场景 LongformerForQuestionAnswering 作为测试模型。在 STEP0 中我们已对齐了 Tokenizer，所以此处使用自然语言数据作为测试数据。（主要是随机数据如果没有 sep_token 模型会报错...）  

## 2. 流程
```
# 生成classifier权重
cd classifier_weights && python generate_classifier_weights.py
# 生成paddle的前向数据
python pd_forward.py
# 生成torch的前向数据
python pt_forward.py
# 对比生成log
python check_step1.py
```

## 3. 结果
&emsp;&emsp;模型前向数据误差在 forward_diff.log 中查看：
```
[2022/04/13 18:42:38] root INFO: logits: 
[2022/04/13 18:42:38] root INFO: 	mean diff: check passed: True, value: 2.2614703709677997e-07
[2022/04/13 18:42:38] root INFO: diff check passed
```
&emsp;&emsp;阈值设定为2e-6，测试通过。