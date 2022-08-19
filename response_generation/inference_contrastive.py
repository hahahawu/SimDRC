import argparse
import os, warnings, logging, random
import json
import sys
import torch
from tqdm import tqdm
from processor import Processor
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from dataset import ResponseData
from torch.utils.data import DataLoader
from SimCTG.SimCTGEncDec.SimCTGBART.simctgbart import SimCTGBART
from SimCTG.pretraining.simctg import SimCTGPretraining
from collections import OrderedDict

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def inference():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--model_dir", type=str, default="checkpoints")
    arg_parse.add_argument("--model_name", type=str, required=True)
    arg_parse.add_argument("--model_version", type=str, required=True)
    arg_parse.add_argument("--beam_size", type=int, default=None)
    arg_parse.add_argument("--min_decode_length", type=int, default=10)
    arg_parse.add_argument("--max_decode_length", type=int, default=128)
    arg_parse.add_argument("--test_batch_size", type=int, default=32)
    arg_parse.add_argument("--test_path", type=str, default="data/dailydialog/dailydialog.test.txt")
    arg_parse.add_argument("--no_cuda", action="store_true", default=False)
    arg_parse.add_argument("--seed", type=int, default=42)
    arg_parse.add_argument("--top_p", type=float, default=None)
    arg_parse.add_argument("--verbose", action="store_true", default=False)
    arg_parse.add_argument("--type", type=str, default="contrastive")

    args = arg_parse.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
    logger.info(" ** ** * device: {} n_gpu: {} * ** ** ".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.seed)

    model_dir = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(model_dir, args.model_version)
    training_config = json.load(open(os.path.join(model_dir, "training.config"), "r", encoding="utf-8"))
    logger.info(" ** ** * training configuration * ** ** ")
    logger.info(training_config)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_config = AutoConfig.from_pretrained(training_config['language_model'])
    model_config.vocab_size = training_config["vocab_size"]
    if "bart" in training_config['language_model']:
        model = SimCTGBART(training_config['language_model'])
    else:
        model = SimCTGPretraining(training_config['language_model'])
    model.model.resize_token_embeddings(len(tokenizer))

    # load pretrained model
    logger.info(" ** ** * Loading model weights from {} * ** ** ".format(model_path))
    state_dict = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_dict[f"model.{k}"] = v
    model.load_state_dict(new_dict)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if hasattr(model, 'module'):
        model = model.module
    model.to(device)
    eou_token = training_config['eou_token']
    test_set = ResponseData(args.test_path, eou=eou_token)
    processor = Processor(tokenizer=tokenizer, max_len=training_config['max_seq_len'], eou=eou_token,
                          model_type=training_config['language_model'], is_training=False,
                          use_cross_loss=False, use_locality_loss=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                             collate_fn=processor.batch_collate_fn)

    logger.info("******* Running training *******")
    logger.info("   NUM examples = {}   ".format(len(test_set)))
    logger.info("   MAX decode length = {}  ".format(args.max_decode_length))
    logger.info("   Test batch size= {}  ".format(args.test_batch_size))

    output_dir = os.path.join("output", args.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model.eval()
    with torch.no_grad():
        with open(os.path.join(output_dir, "{}-max_len-{}-min_len-{}.txt".format(
                args.type, args.max_decode_length, args.min_decode_length)),
                  "w", encoding="utf-8") as fw:
            for step, batch in enumerate(tqdm(test_loader)):
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                eos_token_id = tokenizer.sep_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
                bos_token_id = tokenizer.cls_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
                dids = torch.LongTensor([eos_token_id, bos_token_id]).unsqueeze(0).to(device)
                for i in range(batch["input_ids"].shape[0]):
                    ids = batch["input_ids"][i].unsqueeze(0)
                    if "bart" in training_config['language_model']:
                        response = model.fast_contrastive_search(input_ids=ids, beam_width=5, alpha=0.6,
                                                                 decoding_len=args.max_decode_length, decoder_ids=dids)
                        outputs = []
                        for idx in response:
                            if idx == eos_token_id:
                                break
                            else:
                                outputs.append(idx)
                    else:
                        outputs = model.fast_contrastive_search(input_ids=ids, beam_width=5, alpha=0.6,
                                                                decoding_len=args.max_decode_length)
                    generated = tokenizer.decode(outputs, skip_special_tokens=True)
                    original_history = tokenizer.decode(ids[0], skip_special_tokens=True)
                    labels = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
                    # assert len(generated) == len(original_history) == len(labels)
                    # for i, (word, gold, pred) in enumerate(zip(original_history, labels, generated)):
                    if args.verbose:
                        print(generated)
                    fw.write(json.dumps(
                        {"id": i, "history": original_history, "predict": generated, "label": labels},
                        ensure_ascii=False) + "\n")


if __name__ == '__main__':
    inference()
