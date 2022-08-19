import sys
sys.path.append("bleurtScore")
from bleurt import score
import argparse
import json


if __name__ == '__main__':
    checkpoints = "bleurtScore/bleurt/BLEURT-20"
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    args = parser.parse_args()
    scorer = score.BleurtScorer(checkpoints)

    models = ["bart", "simctg", "simdrc"]
    strategies = ["greedy", "beam", "nucleus", "contrastive"]
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
            score = scorer.score(
                candidates=lines_dict[f"{model}_{strategy}"],
                references=labels,
                batch_size=64
            )
            print(f"{model}_{strategy} - score: {sum(score) / len(score)}\n")
