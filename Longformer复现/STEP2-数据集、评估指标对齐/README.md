# STEP2-数据集、评估指标对齐
## 1. 引言
&emsp;&emsp;本部分我们将进行所需数据集以及评价指标的对齐。包含：WikiHop，TriviaQA，F1 与 ACC 指标。  
&emsp;&emsp;Wikihop 数据集是机器阅读理解领域的经典数据集，每条数据都包含多篇支撑文档、多个选项以及唯一答案。此处将从[官网原始数据](http://qangaroo.cs.ucl.ac.uk/) 进行处理并进行对齐。为了简便考虑，只处理测试集的数据。  
&emsp;&emsp;TriviaQA 也属于机器阅读理解领域，此处数据来源于 [官网](http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz)。由于其原始数据结构不是特别清晰，我们先进行数据的预处理（本部分不涉及其他深度学习框架），再判断 paddle 与 pytorch 的 Dataset 和 Dataloader 是否对齐。  

## 2. WikiHop
&emsp;&emsp;比对了前五个 batch 的数据。
```
python compare_Wikihop.py
```
&emsp;&emsp;结果在 wikihop_diff.log 中查看：  
```
[2022/04/13 23:07:36] root INFO: dataloader_0_0: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_0_1: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_0_2: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_0_3: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_1_0: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_1_1: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_1_2: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_1_3: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_2_0: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_2_1: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_2_2: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_2_3: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_3_0: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_3_1: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_3_2: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_3_3: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_4_0: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_4_1: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_4_2: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: dataloader_4_3: 
[2022/04/13 23:07:36] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/13 23:07:36] root INFO: diff check passed
```

## 3. TriviaQA


## 4. ACC & F1
``` python
# 生成评估指标数据
python generate_metric.py
# 生成误差 log
python compare_metric.py
```
&emsp;&emsp;可见两个ACC算子与F1算子得到的结果完全一致。
``` python
[2022/04/14 12:33:30] root INFO: accuracy: 
[2022/04/14 12:33:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 12:33:30] root INFO: diff check passed
[2022/04/14 12:33:30] root INFO: f1: 
[2022/04/14 12:33:30] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 12:33:30] root INFO: diff check passed
```
