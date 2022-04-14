import torch
import numpy as np
from reprod_log import ReprodLogger
from transformers.models.longformer.tokenization_longformer import LongformerTokenizer
from transformers.models.longformer.modeling_longformer import LongformerForQuestionAnswering
from transformers.models.longformer.configuration_longformer import LongformerConfig

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()
    path = "/root/autodl-tmp/models/longformer-base-4096"

    # prepare tokenizer & model
    tokenizer = LongformerTokenizer.from_pretrained(path)
    model = LongformerForQuestionAnswering.from_pretrained(path, return_dict=False)
    classifier_weights = torch.load(
        "classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    # prepare data
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # forward
    out = model(input_ids, attention_mask=attention_mask)[0]  # only compare start logits
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
