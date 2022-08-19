import logging, warnings, os
import random
import json

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from arg_parser import get_parser_args
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from transformers import AutoTokenizer, BertTokenizer
from dataset import ResponseData
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from processor import Processor
from modeling_bart import BartForConditionalGeneration
from modeling_gpt2 import GPT2LMHeadModel
from utils import find_last_ckpt

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def validation(dev_loader, model, device):
    total_dev_loss = 0
    dev_iter_bar = tqdm(dev_loader, desc="Iter (dev loss=X.XXX)")
    for step, batch in enumerate(dev_iter_bar):
        for k in batch.keys():
            batch[k].to(device)
        dev_loss = model(**batch, return_dict=True).loss
        dev_iter_bar.set_description('Iter (dev loss=%5.3f)' % dev_loss.item())
        total_dev_loss += dev_loss.item()
    return total_dev_loss / len(dev_loader)


def main(args, local_rank):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    if local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        logger.info("Initialize process on RANK {}.".format(local_rank))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    seed = args.seed + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if "chinese" in args.language_model:
        tokenizer = BertTokenizer.from_pretrained(args.language_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    end_of_utterance = '[eou]'
    tokenizer.add_tokens([end_of_utterance])
    tokenizer.pad_token = tokenizer.eos_token
    model_dir = os.path.join(args.output_dir, args.model_name)
    tokenizer.save_pretrained(model_dir)
    if local_rank == 0:
        dist.barrier()

    print("Loading Train Dataset from {} on RANK {}".format(args.train_path, local_rank))
    train_set = ResponseData(args.train_path, eou=end_of_utterance)
    dev_set = ResponseData(args.dev_path, eou=end_of_utterance)
    if local_rank == -1:
        train_sampler = RandomSampler(train_set, replacement=False)
        _batch_size = args.train_batch_size
    else:
        train_sampler = DistributedSampler(train_set)
        _batch_size = args.train_batch_size // args.world_size
    train_processor = Processor(tokenizer=tokenizer, max_len=args.max_seq_len, eou=end_of_utterance,
                                model_type=args.language_model, use_locality_loss=args.locality_loss,
                                use_cross_loss=args.cross_loss)
    dev_processor = Processor(tokenizer=tokenizer, max_len=args.max_seq_len, eou=end_of_utterance,
                              model_type=args.language_model, use_locality_loss=False, use_cross_loss=False)

    train_loader = DataLoader(train_set, batch_size=_batch_size, sampler=train_sampler, num_workers=0,
                              collate_fn=train_processor.batch_collate_fn, pin_memory=True, shuffle=False)
    dev_loader = DataLoader(dev_set, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=dev_processor.batch_collate_fn)

    amp_handle = None
    if args.fp16:
        try:
            from apex import amp
            amp.init(enable_caching=True)
            logger.info("enable fp16 with amp")
        except ImportError:
            raise ImportError("No module named apex.")

    # reserving from the checkpoint
    # loading checkpoints
    if local_rank not in [-1, 0]:
        dist.barrier()

    # model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.language_model)
    if "bart" in args.language_model.lower():
        model = BartForConditionalGeneration.from_pretrained(args.language_model)
    elif "gpt" in args.language_model.lower():
        model = GPT2LMHeadModel.from_pretrained(args.language_model)
    else:
        raise ValueError(f"Unexpected model name {args.language_model}.")
    model.resize_token_embeddings(len(tokenizer))
    # saving configurations
    _train_config = {
        "language_model": args.language_model,
        "vocab_size": len(tokenizer),
        "max_seq_len": args.max_seq_len,
        "dropout": args.dropout,
        "eou_token": end_of_utterance
    }
    json.dump(_train_config, open(os.path.join(model_dir, "training.config"), "w", encoding="utf-8"), indent=2)

    global_step, start_epoch = 0, 0
    best_loss = 10000
    no_improvement = 0
    optim_state_dict = None
    if args.resume_from_ckpt is not None:
        ckpt_model_dir = os.path.join("checkpoints", args.resume_from_ckpt)
        last_check_ckpt = find_last_ckpt(ckpt_model_dir)
        logger.info(f" ***** Resume the checkpoint from {last_check_ckpt} ***** ")
        optim_saved_dict = torch.load(os.path.join(ckpt_model_dir, "optim.bin"))
        global_step, start_epoch = optim_saved_dict["global_step"], optim_saved_dict["epoch"]
        best_loss, no_improvement = optim_saved_dict["best_loss"], optim_saved_dict["no_improvement"]
        optim_state_dict = optim_saved_dict["optimizer"]
        model.load_state_dict(torch.load(os.path.join(ckpt_model_dir, last_check_ckpt)))

    if local_rank == 0:
        dist.barrier()

    if args.fp16:
        model.half()
    if local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            # from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(grouped_params,
                              lr=args.lr,
                              bias_correction=False,
                              max_grad_norm=1.0)
        optimizer = FP16_Optimizer(
            optimizer, dynamic_loss_scale=True)
    else:
        optimizer = torch.optim.AdamW(grouped_params, lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    lr_schedule = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                            num_training_steps=args.max_training_steps)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)

    if local_rank in [-1, 0]:
        # calculating the size of model parameters
        logger.info("******* Model Parameters *******")
        logger.info("Total parameters: {:.3f} M".format(sum([p.numel() for p in model.parameters()]) / 1e6))
        logger.info("Trainable parameters: {:.3f} M".format(
            sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6))

        logger.info("******* Running training *******")
        logger.info("   NUM examples = {}   ".format(len(train_set)))
        logger.info("   MAX training steps = {}  ".format(args.max_training_steps))
        logger.info("   Batch size per GPU = {}  ".format(args.train_batch_size))
        logger.info("   Check per steps = {}   ".format(args.check_steps))
        logger.info("   Using locality loss = {}   ".format(args.locality_loss))
        logger.info("   Using cross loss = {}   ".format(args.cross_loss))
        logger.info("   Margin value  = {}   ".format(args.margin))
        logger.info("   Lr  =  {}   ".format(args.lr))
        logger.info("   Dropout  =  {}".format(args.dropout))

    logger.info("******* CUDA.empty_cache() on RANK {} *******".format(local_rank))
    torch.cuda.empty_cache()
    model.train()

    total_loss_in_period = 0
    stop_flag = False

    for i_epoch in trange(start_epoch + 1, int(args.max_epoch) + 1, desc="Epoch", disable=local_rank not in [-1, 0]):
        if stop_flag:
            break
        if local_rank != -1:
            # Enabling shuffling
            train_sampler.set_epoch(i_epoch)
        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(iter_bar):
            for k in batch.keys():
                batch[k].to(device)
            loss = model(**batch, margin=args.margin, alpha=args.alpha, return_dict=True, simctg=args.simctg).loss
            if n_gpu > 1:
                loss = loss.mean()
            iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

            # ensure that accumulated gradients are normalized
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            total_loss_in_period += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if not args.fp16:
                    lr_schedule.step()
                global_step += 1

            if (global_step + 1) % args.check_steps == 0:
                if local_rank in [-1, 0]:
                    avg_training_loss = total_loss_in_period / args.check_steps
                    logger.info(" ** ** * DEV PROCESS in {} steps* ** ** ".format(global_step))
                    # dev process
                    model.eval()
                    with torch.no_grad():
                        dev_loss = validation(dev_loader, model, device)
                        if dev_loss < best_loss:
                            best_loss = dev_loss
                            logger.info("** ** * Saving the best model and optimizer ** ** * ")
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(model_dir, "model.step.{}.loss.{:.3f}.bin".format(
                                global_step + 1, dev_loss) if not args.only_save_best else "best.model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            no_improvement = 0
                        else:
                            no_improvement += 1
                            if not args.only_save_best:
                                logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(model_dir, "model.step.{}.loss.{:.3f}.bin".format(
                                    global_step + 1, dev_loss))
                                torch.save(model_to_save.state_dict(), output_model_file)

                    logger.info(" ** ** *Global Step: {}.  Average training loss {}, dev loss {} in last {} steps. "
                                "Best loss is {}. No improvements in last {} checks. * ** ** "
                                "".format(global_step, avg_training_loss, dev_loss, args.check_steps, best_loss,
                                          no_improvement))

                    total_loss_in_period = 0
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                if (no_improvement == args.patience_steps or global_step >= args.max_training_steps) and local_rank in \
                        [-1, 0]:
                    stop_flag = True
                    if args.output_file is not None:
                        fw = open(args.output_file, "a+", encoding="utf-8")
                        fw.write(json.dumps(
                            {"id": f"margin_{args.margin}_alpha_{args.alpha if args.alpha is not None else -1}_lr_{args.lr}_bsz_{args.train_batch_size}_dropout_{args.dropout}",
                             "margin": args.margin, "alpha": args.alpha, "lr": args.lr, "best_loss": best_loss, "bsz": args.train_batch_size, "dropout": args.dropout},
                            ensure_ascii=False) + "\n")
                    raise ValueError("Stop training.")

            model.train()

        logger.info("Epoch {} finished on Rank {}".format(i_epoch, local_rank))


def init_process(local_rank, args):
    dist.init_process_group(backend=args.backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    main(args, local_rank)


if __name__ == '__main__':
    args = get_parser_args()

    if args.simctg and (args.locality_loss or args.cross_loss):
        raise ValueError("Cannot assign simctg and locality/cross with True at the same time.")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    model_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if args.world_size < 2 or args.no_cuda:
        main(args, -1)
        exit(0)

    mp.spawn(init_process, args=(args,), nprocs=args.world_size)
