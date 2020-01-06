"""
itemwise_model_performance_2019Dec02.py
version 2019Dec02
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


def numbers_for_eye_paper_v2019dec02(rows, curr_item=False, out_name=''):
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


        with open('../evaluation_study_overall_results_difference/auroc_'+out_name+'.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in bootstrapped_auroc_scores]))
        with open('../evaluation_study_overall_results_difference/precision_'+out_name+'.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in bootstrapped_precision_scores]))
        with open('../evaluation_study_overall_results_difference/recall_'+out_name+'.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in bootstrapped_recall_scores]))

    return 0


def calc_array_difference(man_data, eye_data):
	diff_data = []
	for i in range(len(man_data)):
		diff_data.append(float(man_data[i]) - float(eye_data[i]))
	return diff_data


def calc_CI_of_difference(diff_auroc, diff_pre, diff_re):
        sorted_scores = np.array(diff_auroc)
        sorted_scores.sort()
        sorted_precision_scores = np.array(diff_pre)
        sorted_precision_scores.sort()
        sorted_recall_scores = np.array(diff_re)
        sorted_recall_scores.sort()

        lower_auroc = round(sorted_scores[int(0.05 * len(sorted_scores))], 3)
        upper_auroc = round(sorted_scores[int(0.95 * len(sorted_scores))], 3)
        lower_pre = round(sorted_precision_scores[int(0.05 * len(sorted_precision_scores))], 3)
        upper_pre = round(sorted_precision_scores[int(0.95 * len(sorted_precision_scores))], 3)
        lower_recall = round(sorted_recall_scores[int(0.05 * len(sorted_recall_scores))], 3)
        upper_recall = round(sorted_recall_scores[int(0.95 * len(sorted_recall_scores))], 3)
        return [lower_auroc, upper_auroc, lower_pre, upper_pre, lower_recall, upper_recall]

if __name__ == "__main__":
    exp_setup_dir = '../experimental_setup_files/'
    case_order_18 = load_list('../../jmir-2019/complete_feature_files_evaluation_cases/case_order_rows.txt')
    generalized_dirs = ['../../jmir-2019/evaluation_study_models_manual', '../../jmir-2019/evaluation_study_models_gaze']

    items = get_all_items(generalized_dirs)

    # ## do item wise auroc, precision, and recall analyses
    if False:
        curr_type_all = ['man_all', 'eye_all']  # no longer used
        curr_type_adj = ['man_adj', 'eye_adj']
        for i, curr_exp_dir in enumerate(generalized_dirs):
            models_only_rows = load_list(curr_exp_dir + '/data_models_only.txt')
            for q, item in enumerate(items):
                results = numbers_for_eye_paper_v2019dec02(models_only_rows, item, curr_type_adj[i]+'-'+str(q))
                # ^ now used for numbers_for_jmir_paper_v2019dec02


    if False:
        result_dir_base = '../evaluation_study_overall_results_difference/'
        with open(result_dir_base + 'bootstrap_difference_results-itemwise-2019Dec02.txt', 'w') as f:
            f.write('item\ttype(manual-gaze)\tlower_auroc\tupper_auroc\tlower_precision\tupper_precision\tlower_recall\tupper_recall\t\n')

            for q, item in enumerate(items):
                try:
                    man_adj_auroc = load_list(result_dir_base  + 'auroc_man_adj-'+str(q)+'.txt')
                    eye_adj_auroc = load_list(result_dir_base  + 'auroc_eye_adj-'+str(q)+'.txt')
                    diff_adj_auroc = calc_array_difference(man_adj_auroc, eye_adj_auroc)
                    man_adj_pre = load_list(result_dir_base  + 'precision_man_adj-'+str(q)+'.txt')
                    eye_adj_pre = load_list(result_dir_base  + 'precision_eye_adj-'+str(q)+'.txt')
                    diff_adj_pre = calc_array_difference(man_adj_pre, eye_adj_pre)
                    man_adj_re = load_list(result_dir_base  + 'recall_man_adj-'+str(q)+'.txt')
                    eye_adj_re = load_list(result_dir_base  + 'recall_eye_adj-'+str(q)+'.txt')
                    diff_adj_re = calc_array_difference(man_adj_re, eye_adj_re)

                    adj_result = calc_CI_of_difference(diff_adj_auroc, diff_adj_pre, diff_adj_re)

                    f.write(item+'\tadj\t'+'\t'.join([str(x) for x in adj_result])+'\n')
                except:
                    print("skipping item:\t" + item) 
