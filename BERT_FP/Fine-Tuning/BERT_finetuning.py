import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Metrics import Metrics
import logging
from torch.utils.data import RandomSampler
from transformers import AdamW
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
# from modeling_bert import BertForSequenceClassification

from transformers.models.bert.modeling_bert import BertForSequenceClassification


FT_model = {
    'ubuntu': '/mnt/cache/wuhan1/language_models/bert-base-uncased',
    'douban': '/mnt/cache/wuhan1/language_models/bert-base-chinese',
    'e_commerce': '/mnt/cache/wuhan1/language_models/bert-base-chinese'
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label, lenidx, locality_mask, cross_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lenidx = lenidx
        self.locality_mask = locality_mask
        self.cross_mask = cross_mask


class BERTDataset(Dataset):
    def __init__(self, args, train, tokenizer):
        self.train = train
        self.args = args
        self.bert_tokenizer = tokenizer

    def __len__(self):
        return len(self.train['cr'])

    def __getitem__(self, item):
        cur_features = convert_examples_to_features(item, self.train, self.bert_tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.label, dtype=torch.float),
                       torch.tensor(cur_features.lenidx),
                       torch.tensor(cur_features.locality_mask),
                       torch.tensor(cur_features.cross_mask)
                       )

        return cur_tensors


def _truncate_seq_pair(tokens_a, tokens_b, max_length, eou_token_id):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    if len(tokens_a) + len(tokens_b) <= max_length:
        return

    while eou_token_id in tokens_a:
        item = tokens_a.pop(0)
        if item == eou_token_id and len(tokens_a) + len(tokens_b) <= max_length:
            return

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def get_utterance_boundaries(tokenizer, tokens):
    boundaries = [0]
    # denote [CLS] as the first utterance U1; padding is U0
    dialogue_indices = []
    utterance_idx = 1
    for pos, token in enumerate(tokens):
        if token in [tokenizer.cls_token_id, tokenizer.eos_token_id]:
            boundaries.append(pos)
            dialogue_indices.append(utterance_idx)
            utterance_idx += 1
        else:
            dialogue_indices.append(utterance_idx)
    return boundaries


def get_locality_and_cross_mask(boundaries, max_len):
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


def convert_examples_to_features(item, train, bert_tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    ex_index = item
    input_ids = train['cr'][item]

    sep = input_ids.index(bert_tokenizer.sep_token_id)
    context = input_ids[:sep]
    response = input_ids[sep + 1:]
    _truncate_seq_pair(context, response, 253, bert_tokenizer.eos_token_id)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    context_len = len(context)

    input_ids = [bert_tokenizer.cls_token_id] + context + [bert_tokenizer.sep_token_id] + response + [
        bert_tokenizer.sep_token_id]
    segment_ids = [0] * (context_len + 2)  # context
    segment_ids += [1] * (len(input_ids) - context_len - 2)  # #response
    boundaries = get_utterance_boundaries(bert_tokenizer, input_ids)
    locality_mask, cross_mask = get_locality_and_cross_mask(boundaries, 256)

    lenidx = [1 + context_len, len(input_ids) - 1]

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = 256 - len(input_ids)

    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label=train['y'][item],
                             lenidx=lenidx,
                             locality_mask=locality_mask,
                             cross_mask=cross_mask)
    return features


def compute_contrastive_loss(score_matrix, margin):
    """
       margin: predefined margin to push similarity score away
       score_matrix: bsz x seqlen x seqlen; cosine similarity matrix
       input_ids: bsz x seqlen
    """
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2)  # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix  # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)
    cl_loss = torch.mean(loss_matrix)
    return cl_loss


class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        self.bert_config = config_class.from_pretrained(FT_model[args.task], num_labels=1)
        self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task], do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)
        self.bert_model = model_class.from_pretrained(FT_model[args.task], config=self.bert_config)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        """You can load the post-trained checkpoint here."""
        state_dict = torch.load("../FPT/PT_checkpoint/{}/bert.pt".format(args.task), map_location=self.device)
        self.bert_model.bert.load_state_dict(state_dict=state_dict, strict=False)
        self.bert_model = self.bert_model.to(self.device)

    def forward(self):
        raise NotImplementedError

    @staticmethod
    def calculate_locality_loss(score_matrix, margin, mask):
        locality_score = (margin - score_matrix) * mask
        locality_score = F.relu(locality_score)
        locality_loss = torch.sum(locality_score) / torch.sum(mask)
        return locality_loss

    @staticmethod
    def calculate_isotropy_loss(score_matrix, margin, mask):
        cross_score = (score_matrix + margin) * mask
        cross_score = F.relu(cross_score)
        cross_loss = torch.sum(cross_score) / torch.sum(mask)
        return cross_loss

    def train_step(self, i, data):
        with torch.no_grad():
            batch_ids, batch_mask, batch_seg, batch_y, batch_len, locality_mask, cross_mask = [item.to(device=self.device) for item in data]

        self.optimizer.zero_grad()

        output = self.bert_model(batch_ids, batch_mask, batch_seg, output_hidden_states=True)

        logits = torch.sigmoid(output[0])
        loss = self.loss_func(logits.squeeze(), target=batch_y)

        if self.args.margin is not None:
            last_hidden_states = output[1][-1]
            norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
            cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))

            if self.args.simctg:
                cl_loss = compute_contrastive_loss(cosine_scores, self.args.margin)
                loss += cl_loss

            if locality_mask is not None and cross_mask is not None:
                assert cosine_scores.shape == locality_mask.shape
                assert cosine_scores.shape == cross_mask.shape
                assert self.args.alpha is not None
                locality_loss = self.calculate_locality_loss(cosine_scores, self.args.margin, locality_mask)
                cross_loss = self.calculate_isotropy_loss(cosine_scores, self.args.margin, cross_mask)
                loss += self.args.alpha * locality_loss + (1 - self.args.alpha) * cross_loss
            elif locality_mask is not None:
                assert cosine_scores.shape == locality_mask.shape
                locality_loss = self.calculate_locality_loss(cosine_scores, self.args.margin, locality_mask)
                loss += locality_loss
            elif cross_mask is not None:
                assert cosine_scores.shape == cross_mask.shape
                cross_loss = self.calculate_isotropy_loss(cosine_scores, self.args.margin, cross_mask)
                loss += cross_loss

        loss.backward()

        self.optimizer.step()
        # if i % 100 == 0:
        #     print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
        #                                                            batch_y.size(0)))
        return loss

    def fit(self, train, dev):

        if torch.cuda.is_available():
            self.cuda()

        dataset = BERTDataset(self.args, train, self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=2)

        self.loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)

        for epoch in range(self.args.epochs):
            # print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0
            if epoch >= 2 and self.patience >= 2:
                print("Reload the best model...")
                self.load_state_dict(torch.load(self.args.save_path))
                self.adjust_learning_rate()

            self.train()
            for i, data in tqdm(enumerate(dataloader)):
                loss = self.train_step(i, data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(train['y']) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))

            self.evaluate(dev)

            if self.patience >= 4:
                print("Stop training. No improvements in last {} epochs".format(self.patience))
                break

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    def evaluate(self, dev, is_test=False):
        y_pred = self.predict(dev)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, dev['y']):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        if is_test == False and self.args.task != 'ubuntu':
            self.metrics.segment = 2
        else:
            self.metrics.segment = 10
        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:", result[3], "\t",
              "R2:", result[4], "\t",
              "R5:", result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + \
                self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:", self.best_result[3], "\t",
                  "R2:", self.best_result[4], "\t",
                  "R5:", self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1
            print("No improvement in last {} epochs".format(self.patience))

    def predict(self, dev):
        self.eval()
        y_pred = []
        dataset = BERTDataset(self.args, dev, self.bert_tokenizer)
        dataloader = DataLoader(dataset, batch_size=400)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_y, batch_len = (item.to(self.device) for item in data[:5])
            with torch.no_grad():
                output = self.bert_model(batch_ids, batch_mask, batch_seg)
                logits = torch.sigmoid(output[0]).squeeze()

            if i % 100 == 0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0)))
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred

    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available():
            self.cuda()
