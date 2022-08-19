from Metrics import Metrics
import os
import json


def cal_scores():
    score_dicts = []

    for dataset_name in ["douban", "e_commerce", "ubuntu"]:
        for margin in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                file_name = f"score_files/combine_{dataset_name}_margin_{margin}_alpha_{alpha}"
                if not os.path.exists(file_name):
                    print(f"{file_name} NOT exist.")
                    continue
                metrics = Metrics(file_name)
                res = metrics.evaluate_all_metrics()
                map, mrr, p1, r1, r2, r5 = res
                total_score = sum(res) / len(res)
                _tmp_dict = {"id": file_name, "overall": total_score, "map": map, "mrr": mrr, "p1": p1, "r1": r1,
                             "r2": r2, "r5": r5}
                score_dicts.append(_tmp_dict)
    score_dicts = sorted(score_dicts, key=lambda x: x["overall"])
    with open("ranked_scores.txt", "w", encoding="utf-8") as fw:
        for line in score_dicts:
            fw.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    cal_scores()
