# STEP3-损失函数对齐
&emsp;&emsp;在论文与原代码中，对于 WikiHop 使用的是经典 CrossEntropyLoss，分别计算 start 与 end 的 loss 再求平均。该函数已经在很多复现方案中对齐过了，所以此处不再赘述。本部分主要关注 TriviaQA 中使用的所谓 ***regular CrossEntropyLoss***。  
```
[2022/04/14 15:01:12] root INFO: loss: 
[2022/04/14 15:01:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:01:12] root INFO: diff check passed
```
&emsp;&emsp;由于 paddle 的 gather API 与 torch 有些许区别，此处实现了参数与 torch.gather 相同的 torch 风格的 gather。  