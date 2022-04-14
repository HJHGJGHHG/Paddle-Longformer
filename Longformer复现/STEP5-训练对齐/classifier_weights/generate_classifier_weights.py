import numpy as np
import paddle
import torch


def generate(seed):
    np.random.seed(seed)
    weight = np.random.normal(0, 0.02, (768, 1)).astype("float32")  # base
    paddle_weights = {
        "answer_score.weight": weight,
    }
    torch_weights = {
        "answer_score.weight": torch.from_numpy(weight).t(),
    }
    torch.save(torch_weights, "torch_classifier_weights.bin")
    paddle.save(paddle_weights, "paddle_classifier_weights.bin")


if __name__ == "__main__":
    generate(seed=42)
