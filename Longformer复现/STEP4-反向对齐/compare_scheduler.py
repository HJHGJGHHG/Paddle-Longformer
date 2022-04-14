import torch
import numpy as np
from torch.optim import AdamW
from reprod_log import ReprodDiffHelper, ReprodLogger
from paddlenlp.transformers import LinearDecayWithWarmup
from transformers.optimization import get_scheduler as get_hf_scheduler


def test_lr():
    diff_helper = ReprodDiffHelper()
    pd_reprod_logger = ReprodLogger()
    hf_reprod_logger = ReprodLogger()
    lr = 3e-5
    num_warmup_steps = 200
    num_training_steps = 5000
    milestone = [100, 500, 1000, 2000, 4000]
    torch_optimizer = AdamW(torch.nn.Linear(1, 1).parameters(), lr=lr)
    hf_scheduler = get_hf_scheduler(
        name="linear",
        optimizer=torch_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps, )
    pd_scheduler = LinearDecayWithWarmup(
        learning_rate=lr,
        total_steps=num_training_steps,
        warmup=num_warmup_steps
    )
    
    for i in range(num_training_steps):
        hf_scheduler.step()
        pd_scheduler.step()
        if i in milestone:
            hf_reprod_logger.add(
                f"step_{i}_lr",
                np.array([hf_scheduler.get_last_lr()[-1]]), )
            pd_reprod_logger.add(f"step_{i}_lr",
                                 np.array([pd_scheduler.get_lr()]))
    
    diff_helper.compare_info(hf_reprod_logger.data, pd_reprod_logger.data)
    diff_helper.report(path="scheduler_diff.log")


if __name__ == "__main__":
    test_lr()
