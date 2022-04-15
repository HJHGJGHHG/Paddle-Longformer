# STEP5-训练对齐
## 1. 引言
&emsp;&emsp;在本节中，我们期望对齐训练过程与结果，并最终完成复现。在训练对齐过程中，受到较多随机量的影响，精度有少量 diff 是正常的，diff 在 0.15% 以内可以认为是正常的。  
&emsp;&emsp;值得注意的是，受硬件资源限制，***本部分只关注两种框架的 Longformer-base 模型在 WikiHop 任务下的训练对齐结果***。鉴于本部分的核心任务是对齐而不是寻求复现指标，此处 epoch 取 1，其他超参数相同，如下：  
* batch_size = 1
* gradient_accumulation_steps = 8
* warmup_steps = 200
* lr = 3e-5
* weight_decay = 1e-2
* beta2 = 0.98
* seed = 42

## 2. 训练对齐结果
&emsp;&emsp;分别执行以下指令进行训练：  
```
cd STEP5-训练对齐/torch_train
python train_eval.py
cd ../paddle_train & python train_eval.py
```

&emsp;&emsp;在 1 片 A40（48G）上单精训练约两个半小时，得到结果：  
```


```
