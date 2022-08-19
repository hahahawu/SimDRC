import argparse


def get_parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)

    # loss
    parser.add_argument("--locality_loss", action="store_true", default=False)
    parser.add_argument("--cross_loss", action="store_true", default=False)
    parser.add_argument("--simctg", action="store_true", default=False)
    parser.add_argument("--margin", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.5)

    # model architecture
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--language_model", type=str, default="bert-base-uncased")

    # IO
    parser.add_argument("--train_path", type=str, default="data/persona-chat.train.txt")
    parser.add_argument("--dev_path", type=str, default="data/persona-chat.dev.txt")
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    # training hyper-parameters
    parser.add_argument("--max_training_steps", type=int, default=50000)
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--check_steps", type=int, default=-1)
    parser.add_argument("--patience_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dev_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--only_save_best", action="store_true", default=False)

    # distributed training
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--world_size", type=int, default=-1)
    parser.add_argument("--start_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl", choices=['gloo', 'nccl'])

    return parser.parse_args()
