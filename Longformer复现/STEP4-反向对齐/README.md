# STEP4-反向对齐
&emsp;&emsp;在本节中，我们期望对齐 Scheduler、Optimizer、正则化策略以及最终实现两个模型反向传播的对齐。

## LR Scheduler对齐
&emsp;&emsp;NOTE:本部分Copy自：https://github.com/JunnYu/BERT-SST2-Prod/blob/main/pipeline/Step4/test_lr_scheduler.py
```
python compare_schedular.py
```
&emsp;&emsp;相关误差见：scheduler_diff.log。由于论文只是用了 Linear LR Scheduler with warm-up，所以此处只进行该方法的比较。  
```
[2022/04/14 15:51:12] root INFO: step_100_lr: 
[2022/04/14 15:51:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:51:12] root INFO: step_500_lr: 
[2022/04/14 15:51:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:51:12] root INFO: step_1000_lr: 
[2022/04/14 15:51:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:51:12] root INFO: step_2000_lr: 
[2022/04/14 15:51:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:51:12] root INFO: step_4000_lr: 
[2022/04/14 15:51:12] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 15:51:12] root INFO: diff check passed
```

## 反向对齐
&emsp;&emsp;本部分分别测试了 pytorch 与 paddle 模型在 eval 状态（即关闭随机因子如Dropout）下 10 次反向传播下 Loss 的误差。见 bp_align_diff.log：  
```
python check_step4.py
```
&emsp;&emsp;可见是完全一致的。
```
[2022/04/14 16:10:03] root INFO: loss_0: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_1: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_2: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_3: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_4: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_5: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_6: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_7: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_8: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: loss_9: 
[2022/04/14 16:10:03] root INFO: 	mean diff: check passed: True, value: 0.0
[2022/04/14 16:10:03] root INFO: diff check passed
```
