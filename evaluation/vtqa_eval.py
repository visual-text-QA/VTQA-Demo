import json
import collections


def vtqa_eval_file(pred_json, answer_json, yn_path, lang='zh'):
    yn_answer = json.load(open(yn_path, 'r'))
    pred_data = json.load(open(pred_json, 'r'))
    answer_data = json.load(open(answer_json, 'r'))
    return vtqa_eval(pred_data, answer_data, yn_answer, lang)


def vtqa_eval(pred_data, answer_data, yn_answer, lang='zh'):
    assert len(pred_data) == len(answer_data), 'data length not match'
    pred_dict = {}
    for d in pred_data:
        pred_dict[d['qid']] = d
    answer_dict = {}
    for d in answer_data:
        answer_dict[d['qid']] = d

    (
        em_right,
        yn_right,
        yn_em,
        yn_num,
        extract_f1,
        e_em,
        extract_num,
        generate_f1,
        g_em,
        generate_num,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for k in pred_dict.keys():
        p = pred_dict[k]
        gt = answer_dict[k]
        p['answer'] = p['answer'].lower()
        gt['answer'] = gt['answer'].lower()

        if gt['answer'] == p['answer']:
            em_right += 1

        if gt['answer_type'] == 'YN':
            yn_num += 1
            if gt['answer'] == p['answer']:
                yn_em += 1
            if p['answer'] in yn_answer['yes'] and gt['yes_or_no'] == 'yes':
                yn_right += 1
            elif p['answer'] in yn_answer['no'] and gt['yes_or_no'] == 'no':
                yn_right += 1
        elif gt['answer_type'] == 'E':
            if gt['answer'] == p['answer']:
                e_em += 1
            extract_num += 1
            extract_f1 += maf1(p['answer'], gt['answer'])
        elif gt['answer_type'] == 'G':
            if gt['answer'] == p['answer']:
                g_em += 1
            generate_num += 1
            generate_f1 += maf1(p['answer'], gt['answer'])
    em_all_acc = em_right * 1.0 / len(pred_data)
    print('all answer num: {}, exact match acc: {}'.format(len(pred_data), em_all_acc))

    yn_em_acc = yn_em * 1.0 / yn_num if yn_num != 0 else 0
    yn_acc = yn_right * 1.0 / yn_num if yn_num != 0 else 0
    print(
        'YN answer num: {}, YN acc: {}, YN EM: {}'.format(
            yn_num,
            yn_acc,
            yn_em_acc,
        )
    )

    e_em_acc = e_em * 1.0 / extract_num if extract_num != 0 else 0
    e_f1 = extract_f1 * 1.0 / extract_num if extract_num != 0 else 0
    print(
        'Extract answer num: {}, E F1: {}, E EM: {}'.format(
            extract_num,
            e_f1,
            e_em_acc,
        )
    )

    g_em_acc = g_em * 1.0 / generate_num if generate_num != 0 else 0
    g_f1 = generate_f1 * 1.0 / generate_num if generate_num != 0 else 0
    print(
        'Generate answer num: {}, G F1: {}, G EM: {}'.format(
            generate_num,
            g_f1,
            g_em_acc,
        )
    )
    return em_all_acc, yn_acc, yn_em_acc, e_f1, e_em_acc, g_f1, g_em_acc


def maf1(pred, gt):
    pred_word = [i for i in pred]
    gt_word = [i for i in gt]
    common = collections.Counter(gt_word) & collections.Counter(pred_word)
    num_same = sum(common.values())
    if len(gt_word) == 0 or len(pred_word) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gt_word == pred_word)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_word)
    recall = 1.0 * num_same / len(gt_word)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
