import bert_score
import json
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    models = ["bart", "simctg", "simdrc"]
    strategies = ["beam", "greedy", "nucleus", "contrastive"]
    lines_dict = {}
    for model in models:
        for strategy in strategies:
            lines_dict[f"{model}_{strategy}"] = []
    labels = []

    with open(args.file_name, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            labels.append(jo["label"])
            for model in models:
                for strategy in strategies:
                    lines_dict[f"{model}_{strategy}"].append(jo[f"{model}_{strategy}"])

    for model in models:
        for strategy in strategies:
            (p, r, f) = bert_score.score(
                cands=lines_dict[f"{model}_{strategy}"],
                refs=labels,
                idf=True,
                model_type=args.bert_version,
                lang=args.lang,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                batch_size=args.batch_size,
                nthreads=8,
                rescale_with_baseline=True,
                verbose=True
            )
            print(f"{model}_{strategy} - P: {p.mean()}, R: {r.mean()}, F: {f.mean()}\n")
