# Paddle Longformer

## 1.Einsum 有-6左右量级误差
```
import paddle
import numpy as np
import paddle.nn.functional as F
import torch

paddle.set_device("cpu")
q = np.random.uniform(low=0.0, high=1.0, size=[24, 3, 512, 64])
k = np.random.uniform(low=0.0, high=1.0, size=[24, 3, 512, 64])
ppq = paddle.to_tensor(q)
ppk = paddle.to_tensor(k)
ptq = torch.FloatTensor(q)
ptk = torch.FloatTensor(k)

def compare(a, b):
    a = a.detach().numpy()
    b = b.detach().numpy()
    assert a.shape == b.shape
    abs_dif = np.abs(a - b)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


if __name__ == "__main__":
    compare(torch.einsum("bcxd,bcyd->bcxy", (ptq, ptk)), paddle.einsum("bcxd,bcyd->bcxy", ppq, ppk))

```