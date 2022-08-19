import json
import torch
import argparse
from BARTScore.bart_score import BARTScorer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--file_path", type=str, default="data_sample.txt")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bart_score = BARTScorer(device=device, checkpoint=args.lm)

    models = ["bart", "simctg", "simdrc"]
    strategies = ["beam", "greedy", "nucleus", "contrastive"]
    lines_dict = {}
    for model in models:
        for strategy in strategies:
            lines_dict[f"{model}_{strategy}"] = []
    labels = []

    with open(args.file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            labels.append(jo["label"])
            for model in models:
                for strategy in strategies:
                    lines_dict[f"{model}_{strategy}"].append(jo[f"{model}_{strategy}"])

    for model in models:
        for strategy in strategies:
            print(f"calculating {model}_{strategy}")
            score = bart_score.score(
                srcs=lines_dict[f"{model}_{strategy}"],
                tgts=labels,
                batch_size=args.batch_size
            )
            print(f"{model}_{strategy} - score: {sum(score)/len(score)}\n")
