import os
import time
import paddle
import logging
import datetime
import numpy as np
from paddle.metric import Accuracy
from reprod_log import ReprodLogger
from paddle.optimizer import AdamW
from paddlenlp.transformers import LinearDecayWithWarmup

import utils
from preparations import preprocess_data, WikihopQA_Dataset, get_iter, WikihopQAModel, get_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("paddle_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="Paddle WikiHop Training", add_help=add_help)
    parser.add_argument("--data_cache_dir", default="tokenized.json", help="data cache dir.")
    parser.add_argument("--data_path", default="/root/autodl-tmp/data/wikihop/", help="data dir.")
    parser.add_argument("--model_name_or_path", default="/root/autodl-tmp/models/paddle-longformer-base/",
                        help="path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--device", default="gpu", help="device")
    parser.add_argument("--batch_size", default=8, type=int)  # gradient accumulation steps
    parser.add_argument("--max_length", type=int, default=4096,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
    parser.add_argument("--truncate_seq_len", default=1000000000, type=int)
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--workers", default=0, type=int,
                        help="number of data loading workers (default: 0)", )
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="weight decay (default: 1e-2)", dest="weight_decay", )
    parser.add_argument("--lr_scheduler_type", default="linear", help="the scheduler type to use.")
    parser.add_argument("--num_warmup_steps", default=200, type=int,
                        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument("--val-check-interval", type=int, default=5000,
                        help="number of gradient updates between checking validation loss")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output_dir", default="outputs", help="path where to save")
    parser.add_argument("--test_only", help="only test the model", action="store_true", )
    parser.add_argument("--seed", default=42, type=int, help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument("--fp16", action="store_true", help="whether or not mixed precision training")
    
    return parser


def train_model(args, model, train_iter, dev_iter, scaler, optimizer, lr_scheduler, metric):
    print("Start training")
    start_time = time.time()
    losses = []
    for epoch in range(args.num_train_epochs):
        header = "Epoch: [{}]".format(epoch)
        metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
        metric_logger.add_meter(
            "lr", utils.SmoothedValue(
                window_size=1, fmt="{value}"))
        metric_logger.add_meter(
            "instances/s", utils.SmoothedValue(
                window_size=10, fmt="{value}"))
        steps = 0
        for batch in metric_logger.log_every(train_iter, args.print_freq, header):
            model.train()
            batch_start_time = time.time()
            with paddle.amp.auto_cast(
                    enable=scaler is not None,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                loss, _ = model(**batch)
            
            optimizer.clear_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            lr_scheduler.step()
            metric_logger.update(
                loss=loss.item(), lr=lr_scheduler.get_lr())
            metric_logger.meters["instances/s"].update(1 /
                                                       (time.time() - batch_start_time))
            
            if (steps + 1) % args.val_check_interval == 0:
                evaluate_model(args, model, dev_iter, metric)
            steps += 1
    
    # finish training
    acc = evaluate_model(args, model, dev_iter, metric)
    if args.output_dir:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    return acc, losses


def evaluate_model(args, model, dev_iter, metric, print_freq=2000):
    model.eval()
    metric.reset()
    metric_logger = utils.MetricLogger(logger=logger, delimiter="  ")
    header = "Test:"
    with paddle.no_grad():
        for batch in metric_logger.log_every(dev_iter, print_freq, header):
            loss, sum_prediction_scores = model(**batch)
            
            metric_logger.update(loss=loss.item())
            corrects = metric.compute(sum_prediction_scores.argmax(dim=1), batch["answer_index"])
            metric.update(corrects)
    
    acc_global_avg = metric.accumulate()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(" * Accuracy {acc_global_avg:.10f}".format(
        acc_global_avg=acc_global_avg))
    logger.info(" * Accuracy {acc_global_avg:.10f}".format(
        acc_global_avg=acc_global_avg))
    return acc_global_avg


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    print(args)
    logger.info(args)
    scaler = None
    if args.fp16:
        scaler = paddle.amp.GradScaler()
    paddle.set_device(args.device)
    
    if os.path.exists("dev." + args.data_cache_dir) and os.path.exists("train." + args.data_cache_dir):
        print("Loading preprocessed data..")
    else:
        preprocess_data(args)
    
    train_dataset = WikihopQA_Dataset(args, file_dir="train." + args.data_cache_dir)
    dev_dataset = WikihopQA_Dataset(args, file_dir="dev." + args.data_cache_dir)
    train_iter, dev_iter = get_iter(train_dataset, dev_dataset)
    
    print("Creating model")
    model = WikihopQAModel(args)
    # model.resize_token_embeddings(len(args.tokenizer))
    classifier_weights = paddle.load(
        "/root/autodl-tmp/Paddle-Longformer/Longformer复现/STEP5-训练对齐/classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    
    print("Creating lr_scheduler")
    lr_scheduler = LinearDecayWithWarmup(
        learning_rate=args.lr,
        total_steps=args.num_train_epochs * len(train_iter),
        warmup=args.num_warmup_steps
    )
    
    print("Creating optimizer")
    # Split weights in two groups, one with weight decay and the other not.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params,
        beta1=0.9, beta2=0.98
    )
    metric = Accuracy()
    
    acc, losses = train_model(args, model, train_iter, dev_iter, scaler, optimizer, lr_scheduler, metric)
    return acc, losses


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    paddle.seed(args.seed)
    tokenizer = get_tokenizer(args.model_name_or_path)
    args.tokenizer = tokenizer
    acc, losses = main(args)
    
    reprod_logger = ReprodLogger()
    reprod_logger.add("acc", np.array([acc]))
    reprod_logger.save("train_align_benchmark.npy")
    np.save("paddle_losses.npy", np.array(losses))
