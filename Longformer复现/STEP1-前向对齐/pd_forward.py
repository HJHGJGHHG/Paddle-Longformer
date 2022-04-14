import paddle
from reprod_log import ReprodLogger
from tokenizer import LongformerTokenizer
from modeling import LongformerForQuestionAnswering

if __name__ == "__main__":
    paddle.set_device("cpu")
    # def logger
    reprod_logger = ReprodLogger()
    path = "/root/autodl-tmp/models/paddle-longformer-large"
    
    # prepare tokenizer & model
    tokenizer = LongformerTokenizer.from_pretrained(path)
    model = LongformerForQuestionAnswering.from_pretrained(path)
    classifier_weights = paddle.load(
        "classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()
    
    # prepare data
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    encoding = tokenizer.encode(question, text, return_attention_mask=True)
    input_ids = paddle.to_tensor(encoding["input_ids"], dtype=paddle.int64).unsqueeze(0)
    attention_mask = paddle.to_tensor(encoding["attention_mask"], dtype=paddle.int64).unsqueeze(0)
    
    # forward
    out = model(input_ids)[0]  # only compare start logits
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_paddle.npy")
