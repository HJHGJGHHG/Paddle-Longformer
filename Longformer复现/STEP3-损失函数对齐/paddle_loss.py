import paddle
from reprod_log import ReprodLogger
from tokenizer import LongformerTokenizer
from modeling import LongformerForQuestionAnswering


def gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def regular_crossentropyloss(logits, target, ignore_index=-1, dim=-1):
    """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
    assert logits.ndim == 2
    assert target.ndim == 2
    assert logits.shape[0] == target.shape[0]
    
    # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
    # here, the numerator is the sum of a few potential targets, where some of them is the correct answer
    
    # compute a target mask
    target_mask = target == ignore_index
    # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
    masked_target = target * (1 - paddle.cast(target_mask, dtype=paddle.int64))
    # gather logits
    gathered_logits = gather(logits, dim=-1, index=masked_target)
    # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
    gathered_logits[target_mask] = float('-inf')
    
    # each batch is one example
    gathered_logits = gathered_logits.reshape([1, -1])
    logits = logits.reshape([1, -1])
    
    # numerator = log(sum(exp(gathered logits)))
    log_score = paddle.logsumexp(gathered_logits, axis=dim, keepdim=False)
    # denominator = log(sum(exp(logits)))
    log_norm = paddle.logsumexp(logits, axis=dim, keepdim=False)
    
    # compute the loss
    loss = -(log_score - log_norm)
    
    # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
    # remove those from the loss before computing the sum. Use sum instead of mean because
    # it is easier to compute
    return paddle.masked_select(loss, ~paddle.isinf(loss)).sum()


if __name__ == "__main__":
    paddle.set_device("cpu")
    
    # def logger
    reprod_logger = ReprodLogger()
    
    path = "/root/autodl-tmp/models/paddle-longformer-base"
    
    # prepare tokenizer & model
    tokenizer = LongformerTokenizer.from_pretrained(path)
    model = LongformerForQuestionAnswering.from_pretrained(path)
    classifier_weights = paddle.load(
        "classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()
    
    # prepare data
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    start_positions = paddle.to_tensor([14], dtype=paddle.int64).unsqueeze(0)  # nice
    end_positions = paddle.to_tensor([15], dtype=paddle.int64).unsqueeze(0)  # puppet
    encoding = tokenizer.encode(question, text, return_attention_mask=True)
    input_ids = paddle.to_tensor(encoding["input_ids"], dtype=paddle.int64).unsqueeze(0)
    attention_mask = paddle.to_tensor(encoding["attention_mask"], dtype=paddle.int64).unsqueeze(0)
    
    # forward
    start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
    
    start_loss = regular_crossentropyloss(start_logits, start_positions)
    end_loss = regular_crossentropyloss(end_logits, end_positions)
    loss = (start_loss + end_loss) / 2
    
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_paddle.npy")
