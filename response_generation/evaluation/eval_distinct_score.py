import argparse
import json


def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    统计n-gram频率并用dict存储
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1
    return ngram_distinct_count / ngram_total


def calc_distinct(pair_list):
    """
    calc_distinct
    """
    distinct1 = calc_distinct_ngram(pair_list, 1)
    distinct2 = calc_distinct_ngram(pair_list, 2)
    distinct4 = calc_distinct_ngram(pair_list, 4)
    return [distinct1, distinct2, distinct4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str)
    args = parser.parse_args()

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
            print(f"calculating {model}_{strategy}")
            lines = lines_dict[f"{model}_{strategy}"]
            pair_list = []
            for line in lines:
                tokens = line.strip().split()
                pair_list.append((tokens, []))
            dis_1, dis_2, dis_4 = calc_distinct(pair_list)
            print(f"dis_1: {dis_1}, dis_2: {dis_2}, dis_4: {dis_4}")
