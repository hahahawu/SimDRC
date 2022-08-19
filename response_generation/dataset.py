from torch.utils.data import Dataset
import json


def load_data(data_path, eou):
    data = []
    with open(data_path, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            history, tgt = jo["history"], jo["response"]

            history_word_list = []
            for his in history:
                his_words = his.strip()
                if len(his_words) > 0:
                    his_words += f" {eou}"
                    history_word_list.append(his_words)

            tgt_words = tgt.strip()
            data.append((history_word_list, tgt_words))

    print("There are totally {} cases.".format(len(data)))
    return data


class ResponseData(Dataset):
    def __init__(self, data_path, eou='[eou]'):
        super(ResponseData, self).__init__()
        self.instance = load_data(data_path, eou=eou)

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        return self.instance[index]
