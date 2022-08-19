# This script is used to post-process the annotation data
import codecs
import json
import math


def preprocess_turns(turn_words):
    """
    this function is used to rewrite the turns by adding the speakers (human or agent), return the index mapping (from original word index to new word index)
    :param turn_words:
    :return:
    """

    res = []
    idx_mapping = {}
    new_index = -1
    old_index = -1

    # we need to add "B" into the words. This is because in some cases, the first utterance may include the reference to B
    res.append("agent")
    new_index += 1

    for turn_index in range(0, len(turn_words)):
        if turn_index % 2 == 0:
            res.append("human")
        else:
            res.append("agent")
        new_index += 1

        words = turn_words[turn_index]
        for word in words:
            res.append(word)
            old_index += 1
            new_index += 1
            idx_mapping[old_index] = new_index
    return res, idx_mapping


def find_speaker(words, start_index, round):
    cnt = 0
    for t in range(start_index, -1, -1):
        if words[t] in ['human', 'agent']:
            cnt += 1
            if cnt == round:
                return t


def turn_tagging(words):
    tags = [0] * len(words)
    cur_turn_idx = 0
    for idx in range(len(words)):
        word = words[idx]
        if word == "human" or word == "agent":
            cur_turn_idx += 1
        tags[idx] = cur_turn_idx
    return tags


def gen_data(in_path, out_path):
    import tqdm
    missed = 0
    with codecs.open(in_path, 'r', 'utf-8') as fr, codecs.open(out_path, 'w', 'utf-8') as fw:
        res = []
        for line in tqdm.tqdm(fr):
            jo = json.loads(line.strip())
            sent = jo["sent"]
            srls = jo["srl"]
            sent_id = jo["sentid"]

            if sent.endswith("<SEP>"):
                turns = sent.split("<SEP>")[:-1]  # the last part is empty
            else:
                turns = sent.split("<SEP>")

            turn_words = []
            for turn in turns:
                turn_words.append(turn.strip().split(" "))

            words, idx_mapping = preprocess_turns(turn_words)

            dialog_turns = turn_tagging(words)

            for srl in srls:
                pred_idx = srl["pred"]
                if pred_idx == -1:
                    continue
                if pred_idx not in idx_mapping:
                    missed += 1
                    continue
                pred_idx = idx_mapping[pred_idx]
                args = srl["args"]

                # we need to cut the words so that the later turns would not be taken as input
                input_len = -1
                for _ in range(pred_idx + 1, len(words)):
                    if words[_] == "human" or words[_] == "agent":
                        input_len = _
                        break
                if input_len == -1:
                    input_len = len(words)
                tags = ['O'] * input_len
                tags[pred_idx] = 'B-V'

                is_valid_annotation = True
                for role in args:
                    left_idx, right_idx = args[role]

                    # find the new left_idx
                    if left_idx == -2:
                        # TODO: consider the argument [-2,-1]
                        left_idx = right_idx
                    if left_idx != -1 and left_idx != -2:
                        if left_idx not in idx_mapping:
                            is_valid_annotation = False
                            break
                        left_idx = idx_mapping[left_idx]
                    else:
                        left_idx = find_speaker(words, pred_idx - 1, abs(left_idx))

                    # find the new right_idx
                    if right_idx != -1 and right_idx != -2:
                        if right_idx not in idx_mapping:
                            is_valid_annotation = False
                            break
                        right_idx = idx_mapping[right_idx]
                    else:
                        right_idx = find_speaker(words, pred_idx - 1, abs(right_idx))

                    if left_idx is None or right_idx is None or left_idx >= len(tags) or right_idx >= len(tags):
                        is_valid_annotation = False
                        break

                    # label the tag
                    tags[left_idx] = "B-" + role.upper()
                    for _ in range(left_idx + 1, right_idx + 1):
                        tags[_] = "I-" + role.upper()

                cur_turn = dialog_turns[pred_idx]
                dialog_turns_tmp = [str(cur_turn - dialog_turns[_]) for _ in range(input_len)]

                assert len(dialog_turns_tmp) == input_len

                if is_valid_annotation:
                    res.append((sent_id, pred_idx, words[:input_len], dialog_turns_tmp, tags, words))

                    output_str = str(sent_id) + "|||" + str(pred_idx) + "|||" + " ".join(
                        words[:input_len]) + "|||" + " ".join(dialog_turns_tmp) + "|||" + " ".join(tags)
                    fw.write(output_str + "\n")

                else:
                    missed += 1

        print("Ignore {} instances due to the incorrect annotation ...".format(missed))


if __name__ == "__main__":
    # gen_data("data/train.txt", "model_data/duconv.train.data")
    # gen_data("data/dev.txt", "model_data/duconv.dev.data")
    # for i in range(1, 10):
    #     gen_data("/Users/hanwu/Downloads/processed_json/train.processed.{}.txt".format(i), "../model_data/douban.train.data.{}".format(i))
    # with open("../model_data/douban.train.data.0", "r", encoding="utf-8") as fr, open("../model_data/douban.train.data.0.partial", "w", encoding="utf-8") as fw:
    #     lines = fr.read().split("\n")
    #     for i in range(100000):
    #         fw.write(lines[i] + "\n")
    # n_folds = 10
    # with open("/Users/hanwu/Downloads/processed_json/train.processed.txt", "r", encoding="utf-8") as fr:
    #     lines = fr.read().split("\n")[:-1]
    #     cnt = 0
    #     for i in range(n_folds):
    #         fw = open("/Users/hanwu/Downloads/processed_json/train.processed.{}.txt".format(i), "w", encoding="utf-8")
    #         j = 0
    #         while j < math.floor(len(lines) / n_folds) and cnt < len(lines):
    #             fw.write(lines[cnt] + "\n")
    #             cnt += 1
    #             j += 1
    #         print("{} file processed!".format(i))
    import pickle
    data = pickle.load(open("IEMOCAP_features.pkl", "rb"), encoding="latin1")
    print(len(data))
