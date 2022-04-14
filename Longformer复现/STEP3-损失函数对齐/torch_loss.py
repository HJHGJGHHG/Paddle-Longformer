import torch
from reprod_log import ReprodLogger
from transformers.models.longformer.tokenization_longformer import LongformerTokenizer
from transformers.models.longformer.modeling_longformer import LongformerForQuestionAnswering


def regular_crossentropyloss(logits, target, ignore_index=-1, dim=-1):
    """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
    assert logits.ndim == 2
    assert target.ndim == 2
    assert logits.size(0) == target.size(0)
    
    # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
    # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
    
    # compute a target mask
    target_mask = target == ignore_index
    # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
    masked_target = target * (1 - target_mask.long())
    # gather logits
    gathered_logits = logits.gather(dim=dim, index=masked_target)
    # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
    gathered_logits[target_mask] = float('-inf')
    
    # each batch is one example
    gathered_logits = gathered_logits.view(1, -1)
    logits = logits.view(1, -1)
    
    # numerator = log(sum(exp(gathered logits)))
    log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
    # denominator = log(sum(exp(logits)))
    log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)
    
    # compute the loss
    loss = -(log_score - log_norm)
    
    # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
    # remove those from the loss before computing the sum. Use sum instead of mean because
    # it is easier to compute
    return loss[~torch.isinf(loss)].sum()


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
    start_positions = torch.LongTensor([14]).unsqueeze(0)  # nice
    end_positions = torch.LongTensor([15]).unsqueeze(0)  # puppet
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # forward
    start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
    start_loss = regular_crossentropyloss(start_logits, start_positions)
    end_loss = regular_crossentropyloss(end_logits, end_positions)
    
    loss = (start_loss + end_loss) / 2
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
