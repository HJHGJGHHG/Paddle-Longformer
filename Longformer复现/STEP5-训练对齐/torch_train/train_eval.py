import os
import gc
import time
import torch
import errno
import pandas
import logging
import datetime
import numpy as np
import torch.distributed as dist

from datasets import load_metric
from reprod_log import ReprodLogger
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.longformer.configuration_longformer import LongformerConfig

from preparations import preprocess_data, set_seed, WikihopQA_Dataset, get_iter, WikihopQAModel, get_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("torch_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="PyTorch WikiHop Training", add_help=add_help)
    parser.add_argument("--data_cache_dir", default="tokenized.json", help="data cache dir.")
    parser.add_argument("--data_path", default="/root/autodl-tmp/data/wikihop/", help="data dir.")
    parser.add_argument("--model_name_or_path", default="/root/autodl-tmp/models/longformer-base-4096/",
                        help="path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--batch_size", default=8, type=int)  # gradient accumulation steps
    parser.add_argument("--max_length", type=int, default=4096,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
    parser.add_argument("--truncate_seq_len", default=1000000000, type=int)
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--lr", default=3e-5, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="weight decay (default: 1e-2)", dest="weight_decay", )
    parser.add_argument("--num_warmup_steps", default=200, type=int,
                        help="number of steps for the warmup in the lr scheduler.", )
    parser.add_argument("--val-check-interval", type=int, default=250,
                        help="number of gradient updates between checking validation loss")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=None, help="path where to save")
    parser.add_argument("--test_only", action="store_true", help="only test the model")
    parser.add_argument("--seed", default=42, type=int, help="a seed for reproducible training.")
    # Mixed precision training parameters
    parser.add_argument("--fp16", action="store_true", help="whether or not mixed precision training")
    
    # for ddp
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--ddp", default=False, type=bool)
    return parser


def train_model(args, model, train_iter, dev_iter, scaler, optimizer, lr_scheduler, metric):
    print("Start training")
    start_time = time.time()
    losses = []
    acc = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        torch.distributed.barrier()
        
        header = "Epoch: [{}]".format(epoch)
        for i, batch in enumerate(train_iter):
            batch_start_time = time.time()
            if args.ddp:
                batch = {k: v.to(args.local_rank) for k, v in batch.items()}
            else:
                batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss, _ = model(**batch)
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            lr_scheduler.step()
            
            if (i + 1) % args.print_freq == 0 and (not args.ddp or (args.ddp and dist.get_rank() == 0)):
                logger.info(header + "  [ {0}/{1} ]  Loss:{2:4f}({5:4f})  lr:{3:e}  instances/s:{4:4f}".format(
                    i + 1,
                    len(train_iter),
                    loss.item(),
                    lr_scheduler.get_last_lr()[-1],
                    1 / (time.time() - batch_start_time),
                    np.mean(losses)
                ))
            print(header + "  [ {0}/{1} ]  Loss:{2:4f}({5:4f})  lr:{3:e}  instances/s:{4:4f}".format(
                i + 1,
                len(train_iter),
                loss.item(),
                lr_scheduler.get_last_lr()[-1],
                1 / (time.time() - batch_start_time),
                np.mean(losses)
            ))
            
            del loss
            del batch
            gc.collect()
        
        # finish training
        torch.cuda.empty_cache()
        if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
            acc, _ = evaluate_model(args, model, dev_iter, metric)
    
    if args.output_dir and (not args.ddp or (args.ddp and dist.get_rank() == 0)):
        torch.save(model.module, "model.pt")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    print("Training time {}".format(total_time_str))
    return acc, losses


def evaluate_model(args, model, dev_iter, metric):
    model.eval()
    header = "Test:"
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(dev_iter):
            if args.ddp:
                batch = {k: v.to(args.local_rank) for k, v in batch.items()}
            else:
                batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, sum_prediction_scores = model(**batch)
            
            losses.append(loss.item())
            metric.add_batch(
                predictions=sum_prediction_scores.argmax(dim=1),
                references=batch["answer_index"].squeeze(0), )
            if (i + 1) % args.print_freq == 0 and (not args.ddp or (args.ddp and dist.get_rank() == 0)):
                logger.info(header + "  [ {0}/{1} ]  Loss:{2:4f}".format(i + 1, len(dev_iter), loss.item()))
                print(header + "  [ {0}/{1} ]  Loss:{2:4f}".format(i + 1, len(dev_iter), loss.item()))
    
    acc_global_avg = metric.compute()["accuracy"]
    if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
        print(" * Accuracy {acc_global_avg:.10f}".format(
            acc_global_avg=acc_global_avg))
        logger.info(" * Accuracy {acc_global_avg:.10f}".format(
            acc_global_avg=acc_global_avg))
    return acc_global_avg, losses


def main(args):
    if args.output_dir:
        mkdir(args.output_dir)
    scaler = None
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    
    if args.ddp:
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    if (args.ddp and torch.distributed.get_rank() == 0) or (not args.ddp):
        print(args)
        logger.info(args)
    
    if os.path.exists("dev." + args.data_cache_dir) and os.path.exists("train." + args.data_cache_dir):
        print("Loading preprocessed data..")
    else:
        preprocess_data(args)
        print("Loading preprocessed data..")
    
    train_dataset = WikihopQA_Dataset(args, file_dir="train." + args.data_cache_dir)
    dev_dataset = WikihopQA_Dataset(args, file_dir="dev." + args.data_cache_dir)
    # train_dataset = dev_dataset  # debug
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, shuffle=False)
    else:
        train_sampler, dev_sampler = None, None
    train_iter, dev_iter = get_iter(train_dataset, dev_dataset, train_sampler, dev_sampler)
    
    print("Creating model")
    config = LongformerConfig.from_pretrained(args.model_name_or_path)
    model = WikihopQAModel.from_pretrained(args.model_name_or_path, args=args, config=config)
    model.resize_token_embeddings(len(args.tokenizer))
    classifier_weights = torch.load(
        "/root/autodl-tmp/Paddle-Longformer/Longformer复现/STEP5-训练对齐/classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    if args.ddp:
        model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model.to(args.device)
    
    print("Creating optimizer")
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay,
         },
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      betas=(0.9, 0.98),
                      eps=1e-6
                      )
    
    print("Creating lr_scheduler")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_iter), )
    
    metric = load_metric("metric.py")
    
    if args.test_only:
        acc, losses = evaluate_model(args, model, dev_iter, metric)
    else:
        acc, losses = train_model(args, model, train_iter, dev_iter, scaler, optimizer, lr_scheduler, metric)
    return acc, losses


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    set_seed(args.seed)
    tokenizer = get_tokenizer(args.model_name_or_path)
    args.tokenizer = tokenizer
    acc, losses = main(args)
    
    reprod_logger = ReprodLogger()
    reprod_logger.add("acc", np.array([acc]))
    reprod_logger.save("train_align_benchmark.npy")
    np.save("torch_losses.npy", np.array(losses))
