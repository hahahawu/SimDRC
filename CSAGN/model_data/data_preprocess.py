import json


def build_json_file(in_file, out_file):
    with open(in_file, "r", encoding="utf-8") as fr, open(out_file, "w", encoding="utf-8") as fw:
        for line in fr:
            content = line.strip().split("|||")
            sent_id = content[0]
            pred_idx = int(content[1])
            words = content[2].split(" ")
            tags = content[4].split(" ")
            assert len(words) == len(tags)

            tokens, pred_ids, bert_word_len, speaker_ids = [], [], [], []
            label_vec = []
            tokens.append("[CLS]")
            pred_ids.append(1)
            bert_word_len.append(1)
            label_vec.append("[CLS]")
            speaker_type = 0
            for i, word in enumerate(words):
                if word in ["human", "agent"]:
                    speaker_ids.append(speaker_type)
                    speaker_type = 1 - speaker_type
                tmp_tokens = bert_tokenizer.tokenize(word)
                if len(tmp_tokens) == 0:
                    continue
                bert_word_len.append(len(tmp_tokens))
                tokens.extend(tmp_tokens)
                label_vec.append(tags[i])
                if tags[i] == 'O':
                    label_vec.extend(["O"] * (len(tmp_tokens) - 1))
                elif len(tmp_tokens) > 1:
                    label_vec.extend(["I" + tags[i][1:]] * (len(tmp_tokens) - 1))
                if i == pred_idx:
                    pred_ids.extend([2] * len(tmp_tokens))
                else:
                    pred_ids.extend([1] * len(tmp_tokens))
            tokens.append("[SEP]")
            pred_ids.append(1)
            bert_word_len.append(1)
            label_vec.append('[SEP]')

            assert len(tokens) == len(pred_ids) == len(label_vec) == sum(bert_word_len)

            tmp_dict = {"sent_id": sent_id, "tokens": tokens, "pred_vec": pred_ids, "speaker_vec": speaker_ids,
                        "label": label_vec, "bert_word_len": bert_word_len}

            fw.write(json.dumps(tmp_dict, ensure_ascii=False) + "\n")


def statistics(file_path):
    import numpy as np
    turns = []
    max_token_nums = []
    multi = []
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            turns.append(jo['turn_num'])
            max_token_nums.append(jo['max_turn_tokens'])
            multi.append(jo["turn_num"] * jo['max_turn_tokens'])
    print("turn: max. {} mean. {} median. {}".format(np.max(turns), np.mean(turns), np.median(turns)))
    print('token: max. {} mean. {} median. {}'.format(np.max(max_token_nums), np.mean(max_token_nums), np.median(max_token_nums)))
    print('multi: max. {} mean. {} media. {}'.format(np.max(multi), np.mean(multi), np.median(multi)))


def check_data(file_path):
    with open(file_path, "r", encoding="utf-8") as fr:
        for l_idx, line in enumerate(fr):
            jo = json.loads(line.strip())
            tokens = jo["tokens"]
            pred_vec = jo["pred_vec"]
            label = jo["label"]
            assert len(tokens) == len(pred_vec) == len(label)

            role_idx = {}
            for i, la in enumerate(label):
                if la in ["B-V", "I-V"]:
                    assert pred_vec[i] == 2
                if la not in ["[CLS]", "[SEP]", "O"]:
                    la = la[2:]
                    if la in role_idx:
                        role_idx[la].append(i)
                    else:
                        role_idx[la] = [i]

            tmp_str = []
            for role in role_idx:
                word = "".join([tokens[_].strip("##") if tokens[_].startswith("##") else tokens[_] for _ in role_idx[role]])
                tmp_str.append("{}: {}".format(role, word))

            print("\t".join(tmp_str))


def len_stat(in_file):
    from transformers import BertTokenizer
    import numpy as np
    bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    lens = []
    with open(in_file, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("|||")
            sent = line[2]
            tokens = bert_tokenizer.tokenize(sent)
            lens.append(len(tokens))
    print("min: {}, max: {}, mean: {}".format(np.min(lens), np.max(lens), np.mean(lens)))


if __name__ == '__main__':
    # bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # bert_tokenizer = None
    # build_json_file("duconv.dev.data", "dev.gcn.json")
    # build_json_file("duconv.train.data", "train.gcn.json")
    # build_json_file("duconv.test.data", "test.gcn.json")
    # check_data("train.gcn.json")
    len_stat("personal.test.data")
