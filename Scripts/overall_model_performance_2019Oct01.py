"""
overall_model_performance_2019Oct01.py
version 2019Oct01
package https://github.com/ajk77/jmir-2019
Created by AndrewJKing.com|@andrewsjourney

Result analysis and bootstrapped confidence intervals. 
Run this file to reproduce results. 
Data files are available to run this as is (after setting conditionals to True in __main__).
To reproduce from scratch, please see jmir-2019/drive_assemble_feature_matrix_2019Sep29.py

---LICENSE---
This file is part of jmir-2019
jmir-2019 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
jmir-2019 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with PatientPy.  If not, see <https://www.gnu.org/licenses/>.
"""
import numpy as np
from numpy.random import randint
from sklearn.metrics import roc_auc_score
from scipy.stats import sem


def load_list(filename):
    ll = []
    with open(filename, 'r') as f:
        for full_line in f:
            line = full_line.rstrip()
            if line:
                ll.append(line)
    return ll


def get_all_items(dirs):
    g_items = []
    for c_dir in dirs:
        items_rows = load_list(c_dir + '/data_all_items.txt')
        for row in items_rows:
            if row[0] == '#':
                continue
            else:
                s_row = row.split('\t')
                if s_row[2] not in g_items:
                    g_items.append(s_row[2])
    return g_items


def load_case_items_present(param1, param2):
    icpd, icup = load_items_present_evaluation(param1, param2)
    cip = {}
    for item in param2:
        for case in icpd[item]:
            if case not in cip:
                cip[case] = []
            cip[case].append(item)
    return cip


def load_items_present_evaluation(items_present_file, item_list):
    item_case_present_dict = {}
    item_case_user_present = {}
    item_indices = []
    file_rows = load_list(items_present_file)
    for idx, file_row in enumerate(file_rows):
        split_file_row = file_row.split(',')
        if idx == 0:
            for item in item_list:
                item_indices.append(split_file_row.index(item))
                item_case_present_dict[item] = {}
                item_case_user_present[item] = {}
        else:
            for idx, item in enumerate(item_list):
                if split_file_row[1] not in item_case_present_dict[item]:
                    item_case_present_dict[item][split_file_row[1]] = 0
                    item_case_user_present[item][split_file_row[1]] = {}
                item_case_present_dict[item][split_file_row[1]] += int(split_file_row[item_indices[idx]])
                item_case_user_present[item][split_file_row[1]][split_file_row[0]] = int(
                    split_file_row[item_indices[idx]])

    # ## calculate the binary item_case_present_dict
    for item in item_list:
        for case in item_case_present_dict[item]:
            if item_case_present_dict[item][case] >= 2:
                item_case_present_dict[item][case] = 1
            else:
                item_case_present_dict[item][case] = 0

    return [item_case_present_dict, item_case_user_present]


def align_targets_2(_exp_setup_dir, _curr_exp_dir, _case_order):
    """
    Generalized version
    Finds the cases_items that were not in target_feature_columns (i.e., no model built)
    """
    def get_case_labelers(curr_case, tar_case_rows, tar_participant_order_rows):
        user_ids = []
        for _idx, cid in enumerate(tar_case_rows):
            if cid == curr_case:
                user_ids.append(tar_participant_order_rows[_idx])
        return user_ids

    target_user_case_rows = load_list(_exp_setup_dir + 'pid_target_case_rows.txt')
    target_case_rows = [x.split('_')[1] for x in target_user_case_rows]
    target_feature_columns = load_list(_exp_setup_dir + 'target_feature_columns.txt')  # all models
    target_participant_order_rows = load_list(_exp_setup_dir + 'target_participant-order_rows.txt')
    target_full_matrix_str_list = load_list(_exp_setup_dir + 'target_full_matrix.txt')
    # ^this is the gold standard. The users and cases for each row are target_case_rows and
    # ...target_participant_order_rows. The columns are target_feature_columns

    # ## build gold standard
    case_user_item_gold_std = {}
    for idx, row in enumerate(target_full_matrix_str_list):
        case_user_item_gold_std[target_case_rows[idx] + target_participant_order_rows[idx]] = {}
        split_row = row.rstrip().split(',')
        for idx2, item in enumerate(target_feature_columns):
            case_user_item_gold_std[target_case_rows[idx] + target_participant_order_rows[idx]][item] = split_row[idx2]

    case_set = list(set(target_case_rows))  # the set of 18 cases
    case_items_present = load_case_items_present(_exp_setup_dir + 'items_present-evaluation.txt', target_feature_columns)

    # ## open out files
    out_file_models_only = open(_curr_exp_dir + '/data_models_only.txt', 'w')
    out_file_all_items = open(_curr_exp_dir + '/data_all_items.txt', 'w')
    out_file_models_only.write('#case\tuser\titem\tgold_std\tpreds\tprobs\n')
    out_file_all_items.write('#case\tuser\titem\tgold_std\tpreds\tprobs\n')

    # ## load predictions and probabilities
    pred_rows = load_list(_curr_exp_dir + '/predictions_out.txt')
    prob_rows = load_list(_curr_exp_dir + '/probabilities_out.txt')
    item_preds = {}
    item_probs = {}
    active_models = []
    for i in range(len(pred_rows)):  # for each item row {row is item,case1prediction,case2prediction,...}
        curr_item = pred_rows[i].split(',')[0]
        item_preds[curr_item] = [x for x in pred_rows[i].split(',')[1:]]  # in curr_dir_case_order
        item_probs[curr_item] = [x for x in prob_rows[i].split(',')[1:]]  # in curr_dir_case_order
        active_models.append(curr_item)

    # ## write out to files
    for case in case_set:  # for each case
        case_pred_idx = _case_order.index(case)
        case_labelers = get_case_labelers(case, target_case_rows, target_participant_order_rows)
        for user_id in case_labelers:  # for each usered that labeled that case
            for item in case_items_present[case]:  # for each item present for the case
                if item in active_models:  # if the item has a model
                    out_line = case + '\t' + user_id + '\t' + item + '\t'
                    out_line += case_user_item_gold_std[case + user_id][item] + '\t'
                    out_line += item_preds[item][case_pred_idx]+'\t'+item_probs[item][case_pred_idx]+'\t\n'
                    out_file_models_only.write(out_line)
                    out_file_all_items.write(out_line)
                else:  # if the item does not have a model
                    out_line = case + '\t' + user_id + '\t' + item +'\t' + case_user_item_gold_std[case + user_id][item]
                    out_line += '\t' + '0' + '\t' + '0' + '\t\n'
                    out_file_all_items.write(out_line)

    out_file_models_only.close()
    out_file_all_items.close()

    return


def calc_auroc(gold_std, probabilities, item='', out_writer=False):
    """
    calculates and returns auroc and n
    """
    assert(len(gold_std) == len(probabilities))

    if len(gold_std) == 0:
        score = 'never_labeled'
    elif 0 not in gold_std:
        score = "all_1"
    elif 1 not in gold_std:
        score = "all_0"
    else:
        score = roc_auc_score(np.array(gold_std), np.array(probabilities))

    if out_writer:
        out_writer.write(item + '\t' + str(score) + '\t' + str(sum(gold_std)) + '\t' + str(len(gold_std)) + '\n')
        return
    else:
        return [score, len(gold_std)]


def calc_precision_and_recall(gold_std, predictions, curr_item='', out_writer=False):
    """
    Calculates and returns [precision, recall, count true postive, length]
    """
    assert (len(gold_std) == len(predictions))

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for i in range(len(gold_std)):
        if gold_std[i] == predictions[i]:
            if gold_std[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if gold_std[i] == 1:
                fn += 1
            else:
                fp += 1

    all_predicted = tp + fp
    all_relevant = tp + fn
    if all_predicted:  # check for non-zero denominator
        precision = round(tp / all_predicted, 3)
    else:
        precision = 0
    if all_relevant:  # check for non-zero denominator
        recall = round(tp / all_relevant, 3)
    else:
        recall = 0

    if out_writer:
        out_writer.write(curr_item + '\t' + str(precision) + '\t' + str(recall) + '\t')
        out_writer.write(str(tp) + '\t' + str(fp) + '\t' + str(fn) + '\t' + str(tn) + '\n')
        return
    else:
        return [precision, recall, tp, fp, fn, tn]


def numbers_for_eye_paper_v2019sep28(rows, curr_item=False):
    gold_std = []
    preds = []
    probs = []

    for row in rows:
        if row[0] == '#':
            continue
        else:
            if curr_item:  # for item wise analysis
                s_row = row.split('\t')
                if curr_item == s_row[2]:
                    gold_std.append(float(s_row[3]))
                    preds.append(float(s_row[4]))
                    probs.append(float(s_row[5]))
            else:
                s_row = row.split('\t')
                gold_std.append(float(s_row[3]))
                preds.append(float(s_row[4]))
                probs.append(float(s_row[5]))

    assert(len(gold_std)==len(probs))

    if len(gold_std) == 0:
        score = 'never_labeled'
    elif 0 not in gold_std:
        score = "all_1"
    elif 1 not in gold_std:
        score = "all_0"
    else:
        gold = np.array(gold_std)
        prob = np.array(probs)
        preds = np.array(preds)

        auroc = roc_auc_score(gold, prob)
        precision, recall, tp, fp, fn, tn = calc_precision_and_recall(gold_std, preds)
        auroc = round(auroc, 3)
        precision = round(precision, 3)
        recall = round(recall, 3)

        n_bootstraps = 10000
        rng_seed = 42  # control reproducibility
        bootstrapped_auroc_scores = []
        bootstrapped_precision_scores = []
        bootstrapped_recall_scores = []

        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, high=len(prob) - 1, size=len(prob), dtype='int')
            if len(np.unique(gold[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(gold[indices], prob[indices])
            bootstrapped_auroc_scores.append(score)
            b_precision, b_recall, b_tp, b_fp, b_fn, b_tn = calc_precision_and_recall(gold[indices], preds[indices])
            bootstrapped_precision_scores.append(b_precision)
            bootstrapped_recall_scores.append(b_recall)

        sorted_scores = np.array(bootstrapped_auroc_scores)
        sorted_scores.sort()
        sorted_precision_scores = np.array(bootstrapped_precision_scores)
        sorted_precision_scores.sort()
        sorted_recall_scores = np.array(bootstrapped_recall_scores)
        sorted_recall_scores.sort()

        lower_auroc = round(sorted_scores[int(0.05 * len(sorted_scores))], 3)
        upper_auroc = round(sorted_scores[int(0.95 * len(sorted_scores))], 3)
        lower_pre = round(sorted_precision_scores[int(0.05 * len(sorted_precision_scores))], 3)
        upper_pre = round(sorted_precision_scores[int(0.95 * len(sorted_precision_scores))], 3)
        lower_recall = round(sorted_recall_scores[int(0.05 * len(sorted_recall_scores))], 3)
        upper_recall = round(sorted_recall_scores[int(0.95 * len(sorted_recall_scores))], 3)

        return [auroc, lower_auroc, upper_auroc, precision, lower_pre, upper_pre, recall, lower_recall, upper_recall, tp, fp, fn, tn]
    return 0


def write_numbers_results(out_f, dir_and_type, result_list):
    def r_s(num):
        return str(round(num, 3))

    out_f.write(dir_and_type + '\t')  # dir and type
    out_f.write('\t'.join([r_s(x) for x in result_list]))
    # ^[auroc, lower_auroc, upper_auroc, precision, lower_pre, upper_pre, recall, lower_recall,
    # ...upper_recall, tp, fp, fn, tn]
    out_f.write('\n')  # new line
    return


if __name__ == "__main__":
    exp_setup_dir = '../experimental_setup_files/'
    case_order_18 = load_list('../complete_feature_files_evaluation_cases/case_order_rows.txt')
    generalized_dirs = ['../evaluation_study_models_manual', '../evaluation_study_models_gaze']

    if False:
        for curr_exp_dir in generalized_dirs:
            print('=====' + curr_exp_dir + '=====')
            align_targets_2(exp_setup_dir, curr_exp_dir, case_order_18)

    # ## do overall auroc, precision, and recall analyses
    if False:
        out_file = open('../overall_models_only_results.txt', 'w')
        out_file.write('#dir\ttype\tauroc\tlower-auroc\tupper-auroc\tprecision\tlower-pre\tupper-pre\t')
        out_file.write('recall\tlower-re\tupper-re\ttp\tfp\tfn\ttn\n')

        for curr_exp_dir in generalized_dirs:
            all_items_rows = load_list(curr_exp_dir + '/data_all_items.txt')
            results = numbers_for_eye_paper_v2019sep28(all_items_rows)
            dir_and_type_str = curr_exp_dir + '\tall_item_rows'
            write_numbers_results(out_file, dir_and_type_str, results)

            models_only_rows = load_list(curr_exp_dir + '/data_models_only.txt')
            results = numbers_for_eye_paper_v2019sep28(models_only_rows)
            dir_and_type_str = curr_exp_dir + '\tmodels_only_rows'
            write_numbers_results(out_file, dir_and_type_str, results)

        out_file.close()

'''
--------info------

> Last run on 2019/10/1 
=====../evaluation_study_models_manual=====
=====../evaluation_study_models_gaze=====
-------------------
'''