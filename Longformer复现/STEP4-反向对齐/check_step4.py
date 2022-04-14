import torch
import paddle
from transformers import AdamW
from reprod_log import ReprodLogger, ReprodDiffHelper
from tokenizer import LongformerTokenizer as PDTokenizer
from modeling import LongformerForQuestionAnswering as PDQA
from transformers.models.longformer.tokenization_longformer import LongformerTokenizer as PTTokenizer
from transformers.models.longformer.modeling_longformer import LongformerForQuestionAnswering as PTQA

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"


def pd_train_some_iters(max_iter=10):
    path = "/root/autodl-tmp/models/paddle-longformer-base"
    
    # prepare tokenizer & model
    tokenizer = PDTokenizer.from_pretrained(path)
    model = PDQA.from_pretrained(path)
    classifier_weights = paddle.load(
        "classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()
    
    # prepare data
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    start_positions = paddle.to_tensor([14], dtype=paddle.int64)  # nice
    end_positions = paddle.to_tensor([15], dtype=paddle.int64)   # puppet
    encoding = tokenizer.encode(question, text, return_attention_mask=True)
    input_ids = paddle.to_tensor(encoding["input_ids"], dtype=paddle.int64).unsqueeze(0)
    attention_mask = paddle.to_tensor(encoding["attention_mask"], dtype=paddle.int64).unsqueeze(0)
    
    criterion = paddle.nn.CrossEntropyLoss()
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=3e-5,
        parameters=model.parameters(),
        weight_decay=1e-2,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params, )
    loss_list = []
    for idx in range(max_iter):
        start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
        
        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)
    return loss_list


def pt_train_some_iters(max_iter=10):
    path = "/root/autodl-tmp/models/longformer-base-4096"
    
    # prepare tokenizer & model
    tokenizer = PTTokenizer.from_pretrained(path)
    model = PTQA.from_pretrained(path, return_dict=False)
    classifier_weights = torch.load(
        "classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    
    # prepare data
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    start_positions = torch.LongTensor([14])  # nice
    end_positions = torch.LongTensor([15])  # puppet
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    criterion = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    
    loss_list = []
    for idx in range(max_iter):
        start_logits, end_logits = model(input_ids, attention_mask=attention_mask)
        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)
    return loss_list


if __name__ == "__main__":
    print("Start training")
    paddle.set_device("cpu")
    
    pt_reprod_logger = ReprodLogger()
    pt_loss_list = pt_train_some_iters(max_iter=10)
    for idx, loss in enumerate(pt_loss_list):
        pt_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    pt_reprod_logger.save("bp_align_torch.npy")
    
    pd_reprod_logger = ReprodLogger()
    pd_loss_list = pt_train_some_iters(max_iter=10)
    for idx, loss in enumerate(pd_loss_list):
        pd_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    pd_reprod_logger.save("bp_align_paddle.npy")
    
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./bp_align_torch.npy")
    paddle_info = diff_helper.load_info("./bp_align_paddle.npy")
    diff_helper.compare_info(torch_info, paddle_info)
    
    diff_helper.report(path="bp_align_diff.log")
