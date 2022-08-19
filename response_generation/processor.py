import torch

from torch.nn.utils.rnn import pad_sequence


class InputFeatExample(object):
    def __init__(self, context, tgt_token, encoder_padding_mask, locality_mask=None, cross_mask=None):
        self.context = torch.as_tensor(context)
        self.tgt_token = torch.as_tensor(tgt_token)
        self.encoder_padding_mask = torch.as_tensor(encoder_padding_mask)
        self.locality_mask = locality_mask
        self.cross_mask = cross_mask


class Processor(object):
    def __init__(self, tokenizer, max_len, eou='[eou]', model_type=None, is_training=True,
                 use_locality_loss=False, use_cross_loss=False):
        super(Processor, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eou_id = tokenizer.convert_tokens_to_ids([eou])[0]
        self.model_type = model_type
        self.is_training = is_training
        self.use_locality_loss = use_locality_loss
        self.use_cross_loss = use_cross_loss

    def truncate_sequence(self, tokens, max_len):
        if len(tokens) <= max_len:
            return

        while self.eou_id in tokens:
            item = tokens.pop(1)
            if item == self.eou_id and len(tokens) <= max_len:
                return

        return tokens[-max_len:]

    def get_utterance_boundaries(self, tokens):
        boundaries = []
        utterance_idx = 0
        if self.tokenizer.eos_token_id:
            boundaries = [0]
            utterance_idx = 1
        # denote [CLS] as the first utterance U1; padding is U0
        dialogue_indices = []
        for pos, token in enumerate(tokens):
            if token in [self.tokenizer.cls_token_id, self.eou_id]:
                boundaries.append(pos)
                dialogue_indices.append(utterance_idx)
                utterance_idx += 1
            else:
                dialogue_indices.append(utterance_idx)
        return boundaries, dialogue_indices, utterance_idx - 1

    @staticmethod
    def matrix_mask_padding(input_mask, max_len):
        for i in range(len(input_mask)):
            _tmp_len = len(input_mask[i])
            for x in range(_tmp_len):
                input_mask[i][x] = input_mask[i][x] + [0 for _ in range(max_len - _tmp_len)]
            while len(input_mask[i]) < max_len:
                input_mask[i].append([0 for _ in range(max_len)])
        return input_mask

    @staticmethod
    def _get_locality_and_cross_mask(boundaries, max_len):
        locality_mask = [[0 for _ in range(max_len)] for _ in range(max_len)]
        for i in range(len(boundaries) - 1):
            s = boundaries[i]
            e = boundaries[i+1]
            for x in range(s, e):
                locality_mask[x][e] = 1
                locality_mask[e][x] = 1
        cross_mask = [[0 for _ in range(max_len)] for _ in range(max_len)]
        for x in boundaries:
            for y in boundaries:
                cross_mask[x][y] = 1 if x != y else 0
        return locality_mask, cross_mask

    def collate_fn_by_case(self, batch):
        history_list, tgt_words = batch

        history_words = " ".join(history_list)
        history_tokens = self.tokenizer.encode(history_words)
        tgt_tokens = self.tokenizer.encode(tgt_words)
        if self.tokenizer.eos_token_id is not None:
            history_tokens.append(self.tokenizer.eos_token_id)
            tgt_tokens.append(self.tokenizer.eos_token_id)

        while len(tgt_tokens) > self.max_len:
            tgt_tokens.pop(-2)

        locality_mask, cross_mask = None, None
        if "bart" in self.model_type:
            self.truncate_sequence(history_tokens, self.max_len)
            boundaries, _, _ = self.get_utterance_boundaries(history_tokens)
            locality_mask, cross_mask = self._get_locality_and_cross_mask(boundaries, len(history_tokens))
        elif self.is_training:  # dialogpt
            self.truncate_sequence(history_tokens, self.max_len - len(tgt_tokens))
            labels = [-100 if self.is_training else self.tokenizer.pad_token_id for _ in range(
                len(history_tokens))] + tgt_tokens
            boundaries, _, _ = self.get_utterance_boundaries(history_tokens)
            history_tokens.extend(tgt_tokens)
            locality_mask, cross_mask = self._get_locality_and_cross_mask(boundaries, len(history_tokens))
            tgt_tokens = labels
        else:
            self.truncate_sequence(history_tokens, self.max_len - len(tgt_tokens))

        attention_mask = [1 for _ in range(len(history_tokens))]
        input_example = InputFeatExample(context=history_tokens,
                                         tgt_token=tgt_tokens,
                                         encoder_padding_mask=attention_mask,
                                         locality_mask=locality_mask,
                                         cross_mask=cross_mask)
        return input_example

    def batch_collate_fn(self, batches):
        batch_sample = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
        if self.use_locality_loss:
            batch_sample["locality_mask"] = []
        if self.use_cross_loss:
            batch_sample["cross_mask"] = []

        for batch in batches:
            example = self.collate_fn_by_case(batch)
            batch_sample["input_ids"].append(example.context)
            batch_sample["labels"].append(example.tgt_token)
            batch_sample["attention_mask"].append(example.encoder_padding_mask)
            if self.use_locality_loss:
                batch_sample["locality_mask"].append(example.locality_mask)
            if self.use_cross_loss:
                batch_sample["cross_mask"].append(example.cross_mask)

        batch_sample["input_ids"] = pad_sequence(batch_sample["input_ids"], batch_first=True,
                                                 padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        batch_sample["labels"] = pad_sequence(batch_sample["labels"], batch_first=True,
                                              padding_value=-100 if self.is_training else self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
        batch_sample["attention_mask"] = pad_sequence(batch_sample["attention_mask"], batch_first=True,
                                                      padding_value=0)

        if self.use_locality_loss:
            batch_sample["locality_mask"] = torch.as_tensor(self.matrix_mask_padding(
                batch_sample["locality_mask"], max_len=batch_sample["input_ids"].shape[1]
            ))
        if self.use_cross_loss:
            batch_sample["cross_mask"] = torch.as_tensor(self.matrix_mask_padding(
                batch_sample["cross_mask"], max_len=batch_sample["input_ids"].shape[1]
            ))

        return batch_sample
