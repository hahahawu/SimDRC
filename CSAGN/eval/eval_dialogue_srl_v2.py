
import os, sys, json
from collections import Counter


def index_of_first_greater(value, lst):
    return [x > value for x in lst].index(True)


def load_data(path, key_type, is_cross_only):
    data = {}
    for line in open(path, 'r'):
        jobj = json.loads(line.strip())
        sentid = jobj['sentid']
        assert sentid not in data
        data[sentid] = []
        conversation = []
        conversation_accum_length = []
        for sent in jobj['sent'].split('<SEP>'):
            sent = sent.strip().split()
            if sent == []:
                continue
            conversation.extend(sent)
            conversation_accum_length.append(len(conversation))
        for pa_structure in jobj['srl']:
            pas = {'V': conversation[pa_structure['pred']]}
            pi = index_of_first_greater(pa_structure['pred'], conversation_accum_length)
            for k, v in pa_structure['args'].items():
                st, ed = v
                if ed == -1:
                    v = '我'
                elif ed == -2:
                    v = '你'
                else:
                    v = ' '.join(conversation[st:ed+1])
                if is_cross_only and \
                        (st >= 0 and index_of_first_greater(st, conversation_accum_length) == pi):
                    continue
                if key_type != None and key_type.lower() != k.lower():
                    continue
                pas[k] = v
            data[sentid].append(pas)
    return data


def update_counts_intersect(v1, v2, is_token_level):
    if v1 == '' or v2 == '':
        return 0
    if is_token_level:
        v1 = Counter(v1.split())
        v2 = Counter(v2.split())
        res = 0
        for k, cnt1 in v1.items():
            if k in v2:
                res += min(cnt1, v2[k])
        return res
    else:
        return v1 == v2


def update_counts_denominator(conv, is_token_level):
    counts = 0
    for pas in conv:
        for k, v in pas.items():
            if k != 'V': # don't count "pred" for each PA structure
                counts += len(v.split()) if is_token_level else 1
    return counts


def update_counts(ref_conv, prd_conv, counts, is_token_level):
    counts[1] += update_counts_denominator(ref_conv, is_token_level)
    counts[2] += update_counts_denominator(prd_conv, is_token_level)
    for ref_pas, prd_pas in zip(ref_conv, prd_conv):
        for k, v1 in ref_pas.items():
            if k == 'V':
                continue
            v2 = prd_pas.get(k,'')
            counts[0] += update_counts_intersect(v1, v2, is_token_level)


def calc_f1(ref, prd, is_token_level):
    """
    :param ref: a list of predicate argument structures
    :param prd:
    :return:
    """
    counts = [0, 0, 0]
    update_counts(ref, prd, counts, is_token_level)
    p = 0.0 if counts[2] == 0 else counts[0]/counts[2]
    r = 0.0 if counts[1] == 0 else counts[0]/counts[1]
    f = 0.0 if p == 0.0 or r == 0.0 else 2*p*r/(p+r)
    return {'P':p, 'R':r, 'F':f}


def eval_main(key_type, is_cross_only, is_token_level):
    ref = load_data("../data/dev.txt", key_type, is_cross_only)
    prd = load_data("../data/dev.txt", key_type, is_cross_only)
    ref_list = []
    prd_list = []
    for key, ref_data in ref.items():
        prd_data = prd.get(key, [])
        ref_list.extend(ref_data)
        prd_list.extend(prd_data)
    print(calc_f1(ref_list, prd_list, is_token_level))


if __name__ == "__main__":
    is_token_level = False
    for is_cross_only in [False, True]:
        if is_cross_only == False:
            print("=====Evaluating whole dataset=====")
            for key_type in ['arg0', 'arg1', 'arg2', 'tmp', 'loc', 'prp']:
                print('-----For {}-----'.format(key_type))
                eval_main(key_type, is_cross_only, is_token_level)
        else:
            print("=====Evaluating only the cross-turn cases=====")
            eval_main(None, is_cross_only, is_token_level)
