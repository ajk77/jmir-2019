"""
instantiate_experiment_2019sep09.py
version 2019sep09
package https://github.com/ajk77/jmir-2020-king
Created by AndrewJKing.com|@andrewsjourney

This file is for cleaning, imputation, feature selection, and model experimentation.
For more details see https://github.com/ajk77/PatientPy

---DEPENDENCIES---
https://github.com/ajk77/PatientPy
https://github.com/ajk77/RegressiveImputer
https://github.com/ajk77/PatientPyFeatureSelection

Must first run PatientPy/patient_pickler.py to store data structures that are used here.
^ set pkl_dir to the same value as was used in patient_pickler.
Must second run PatientPy/create_feature_vectors.py once for each directory filled by create_feature_vectors.py.
Must third run jmir-2020-king/Scripts/driver_assemble_feature_matrix_2019Sep29.py.
Then run this file.

---LICENSE---
This file is part of jmir-2020-king
jmir-2020-king is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
jmir-2020-king is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with PatientPy.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
sys.path.insert(0, '../../../../git_projects/RegressiveImputer')
sys.path.insert(0, '../../../../git_projects/PatientPyFeatureSelection')

from RegressiveImputer import RegressiveImputer, get_clean_columns
from sklearn.preprocessing import Imputer
from RecursiveFeatureInclusion import determine_attribute_sets, staged_feature_inclusion
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
import multiprocessing
import numpy as np
import pickle
import datetime
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def load_list(file_name):
    loaded_list = []
    with open(file_name, 'r') as f:
        for full_line in f:
            line = full_line.rstrip()
            if line:
                loaded_list.append(line)
    return loaded_list


def imputation(params):
    """
    This function runs feature column cleaning and imputation
    """

    # ## load data
    full_training_data = np.load(params['assemble_output_filename'] + '.npy')
    full_training_names = np.load(params['assemble_output_filename'] + '_names.npy')
    full_testing_data = np.load(params['assemble_eval_output_filename'] + '.npy')
    full_testing_names = np.load(params['assemble_eval_output_filename'] + '_names.npy')
    print('Train data loaded: ', full_training_data.shape, full_training_names.shape)
    print('Eval data loaded: ', full_testing_data.shape, full_testing_names.shape)
    print('='*20)

    # ## clean columns to eliminate all nan, only one not nan, and non-uniques
    keep_columns = get_clean_columns(full_training_data)
    cleaned_training_data = full_training_data[:, keep_columns]
    cleaned_training_names = full_training_names[keep_columns, :]
    cleaned_testing_data = full_testing_data[:, keep_columns]
    cleaned_testing_names = full_testing_names[keep_columns, :]
    pickle.dump(keep_columns, open(params['keep_columns_out'], 'wb'))
    print('Train columns cleaned: ', cleaned_training_data.shape, cleaned_training_names.shape)
    print('Eval columns cleaned: ', cleaned_testing_data.shape, cleaned_testing_names.shape)
    print('='*20)

    if params['unit_testing']:
        '''
        This reduces the feature columns for faster runtime and visual inspection
        '''
        keep_columns = [0, 1, 2, 3, 4, 5, 6, 7, 999, 9000]  # enter any desired columns here
        cleaned_training_data = cleaned_training_data[:, keep_columns]
        cleaned_training_names = cleaned_training_names[keep_columns, :]
        cleaned_testing_data = cleaned_testing_data[:, keep_columns]
        cleaned_testing_names = cleaned_testing_names[keep_columns, :]
        print('tests: ', cleaned_training_data.shape, cleaned_training_names.shape)
        print(cleaned_training_data[1:5, 8], cleaned_testing_data[1:5, 8])
        print('='*20)

    # ## use regressive imputer
    r_imputer = RegressiveImputer(params['r_imputer_out'] + '/')
    r_imputed_training_data = r_imputer.fit_transform(np.copy(cleaned_training_data))
    r_imputed_testing_data = r_imputer.transform(np.copy(cleaned_testing_data))
    r_imputed_training_names = r_imputer.transform_column_names(np.copy(cleaned_training_names))
    r_imputed_test_names = r_imputer.transform_column_names(np.copy(cleaned_testing_names))
    pickle.dump(r_imputer, open(params['r_imputer_out'] + '.pkl', 'wb'))
    np.save(params['assemble_output_filename'] + '_rImp', r_imputed_training_data)
    np.save(params['assemble_output_filename'] + '_rImp_names', r_imputed_training_names)
    np.save(params['assemble_eval_output_filename'] + '_rImp', r_imputed_testing_data)
    np.save(params['assemble_eval_output_filename'] + '_rImp_names', r_imputed_test_names)

    print('Train rImp: ', r_imputed_training_data.shape, r_imputed_training_names.shape, r_imputed_training_data.shape[1] == r_imputed_training_names.shape[0])
    print('Eval rImp: ', r_imputed_testing_data.shape, r_imputed_test_names.shape, r_imputed_testing_data.shape[1] == r_imputed_test_names.shape[0])
    # print(r_imputed_training_data[1:5,8], r_imputed_testing_data[1:5,8])
    print('='*20)

    # ## use median imputer
    m_imputer = Imputer(axis=0, missing_values='NaN', strategy='median', verbose=0)  # median deletes first column
    m_imputed_training_data = m_imputer.fit_transform(np.copy(cleaned_training_data))
    m_imputed_testing_data = m_imputer.transform(np.copy(cleaned_testing_data))
    pickle.dump(m_imputer, open(params['m_imputer_out'] + '.pkl', 'wb'))
    np.save(params['assemble_output_filename'] + '_mImp', m_imputed_training_data)
    np.save(params['assemble_output_filename'] + '_mImp_names', cleaned_training_names)
    np.save(params['assemble_eval_output_filename'] + '_mImp', m_imputed_testing_data)
    np.save(params['assemble_eval_output_filename'] + '_mImp_names', cleaned_testing_names)

    print('Train mImp: ', m_imputed_training_data.shape, cleaned_training_names.shape, m_imputed_training_data.shape[1] == cleaned_training_names.shape[0])
    print('Eval mImp: ', m_imputed_testing_data.shape, cleaned_testing_names.shape, m_imputed_testing_data.shape[1] == cleaned_testing_names.shape[0])
    # print(m_imputed_training_data[1:5,8], m_imputed_testing_data[1:5,8])
    print('+'*20)

    return


def populate_imputation_params(type_idx=0):
    f_params = {}
    cases = ['manual_training', 'gaze_training']
    case = cases[type_idx]
    base_dir = '../'
    if case == 'manual_training':  # the manual labeling cases
        f_params['assemble_output_filename'] = base_dir + 'feature_matrix_storage_manual_training_cases/full_manual_training'
        f_params['assemble_eval_output_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation'
        f_params['keep_columns_out'] = base_dir + 'imputer_storage/keep_col_imputer-full_manual_training.pkl'
        f_params['r_imputer_out'] = base_dir + 'imputer_storage/r_imputer-full_manual_training'
        f_params['m_imputer_out'] = base_dir + 'imputer_storage/m_imputer-full_manual_training'
        f_params['unit_testing'] = False
    elif case == 'gaze_training':  # the eye tracking labeling cases
        f_params['assemble_output_filename'] = base_dir + 'feature_matrix_storage_gaze_training_cases/full_gaze_training'
        f_params['assemble_eval_output_filename'] = base_dir + 'feature_matrix_storage_gaze_eval_cases/full_gaze_eval'
        f_params['keep_columns_out'] = base_dir + 'imputer_storage/keep_col_imputer-full_gaze_training.pkl'
        f_params['r_imputer_out'] = base_dir + 'imputer_storage/r_imputer-full_gaze_training'
        f_params['m_imputer_out'] = base_dir + 'imputer_storage/m_imputer-full_gaze_training'
        f_params['unit_testing'] = False

    return f_params


def determine_feature_matrix_and_target_matrix_rows(params):
    """
    This generates a file that stores the followg details:
    feature_matrix_name, target_id, target_name, fold_type [all, 0, 1, 2, 3, 4], [row indices for desired samples]
    The row indices are based on case order rows and item_present-labeling

    It also generates a file that stores the followg details:
    target_matrix_name, target_id, fold_type [all, 0, 1, 2, 3, 4], [row indices for desired samples]
    The row indices are based on target order rows and the rows used in the above file

    It also generates a file that stores the following details:
    feature_matrix_name, target_id, fold_type [all, 0, 1, 2, 3, 4], [case ids of desired samples]
    The row indices are based on the two files above
    """
    def load_target_present_rows(item_present_file, target_name):
        """
        This function determines which cases (samples) a target was present for.
        """
        target_present_rows = []
        with open(item_present_file, 'r') as f:
            target_name_file_column_idx = False
            first_line = True
            for full_line in f:
                line = full_line.rstrip()
                if not line:  # insure line is not empty
                    break
                split_line = full_line.rstrip().split(',')
                if first_line:  # first line
                    first_line = False
                    target_name_file_column_idx = split_line.index(target_name)
                elif int(split_line[target_name_file_column_idx]):  # check if column is '1' for current row
                    target_present_rows.append(split_line[1])  # add case_id
        return target_present_rows

    case_order_rows = load_list(params['case_order_rows_file'])  # the case order in the feature matrix
    target_case_rows = load_list(params['target_case_rows_file'])  # the case order in the target matrix
    target_feature_columns = load_list(params['target_feature_columns_file'])  # the target names for each column in the target matrix

    feature_samples_outfile = open(params['feature_samples_outfile'], 'w')
    target_samples_outfile = open(params['target_samples_outfile'], 'w')
    feat_and_targ_samples_outfile = open(params['feat_targ_samples_outfile'], 'w')
    feature_samples_outfile.write('#feature_matrix_name, target_id, target_name, fold_type, row_indices\n')
    target_samples_outfile.write('#target_matrix_name, target_id, target_name, fold_type, row_indices\n')
    feat_and_targ_samples_outfile.write('#target_matrix_name, target_id, target_name, fold_type, case_ids\n')
    
    for t_idx, target_name in enumerate(target_feature_columns):
        target_present_rows = load_target_present_rows(params['item_present_file'], target_name)
        samples_full_fold_type = []
        target_full_fold_type = []
        feat_targ_full_fold_type = []
        sam_folds = [[] for x in range(5)]  # five folded
        tar_folds = [[] for x in range(5)]  # five folded
        feat_folds = [[] for x in range(5)]  # five filded
        count = 0
        for idx, case_id in enumerate(case_order_rows):
            if case_id in target_present_rows:
                samples_full_fold_type.append(str(idx))
                target_full_fold_type.append(str(target_case_rows.index(case_id)))
                feat_targ_full_fold_type.append(case_id)
                sam_folds[count%5].append(str(idx))
                tar_folds[count%5].append(str(target_case_rows.index(case_id)))
                feat_folds[count%5].append(case_id)
                count += 1

        # ## print the full fold type
        feature_samples_outfile.write(params['feature_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(samples_full_fold_type)+'\n')
        target_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(target_full_fold_type)+'\n')
        feat_and_targ_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+'full'+'\t'+'\t'.join(feat_targ_full_fold_type)+'\n')

        # ## print for the five folds
        for f_idx in range(5):
            feature_samples_outfile.write(params['feature_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(sam_folds[f_idx])+'\n')
            target_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(tar_folds[f_idx])+'\n')
            feat_and_targ_samples_outfile.write(params['target_matrix_name']+'\t'+str(t_idx)+'\t'+target_name+'\t'+str(f_idx)+'\t'+'\t'.join(feat_folds[f_idx])+'\n')

    feature_samples_outfile.close()
    target_samples_outfile.close()
    feat_and_targ_samples_outfile.close()

    return


def populate_sample_rows_params(type_idx=0):
    f_params = {}
    cases = ['manual_training', 'gaze_training']
    case = cases[type_idx]
    base_dir = '../'
    if case == 'manual_training':  # the first four 'burn in' cases
        case_dir = base_dir + 'complete_feature_files_manual_training_cases/'
        out_dir = base_dir + 'feature_matrix_storage_manual_training_cases/'
        f_params['case_order_rows_file'] = case_dir + 'case_order_rows.txt'
        f_params['target_case_rows_file'] = case_dir + 'manual_target_case_rows.txt'
        f_params['target_feature_columns_file'] = case_dir + 'manual_target_feature_columns.txt'
        f_params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
        f_params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
        f_params['feat_targ_samples_outfile'] = out_dir + 'feat_targ_samples_out.txt'
        f_params['item_present_file'] = case_dir + 'items_present-manual_labeling.txt'
        f_params['feature_matrix_name'] = out_dir + 'full_manual_training_'
        f_params['target_matrix_name'] = case_dir + 'manual_target_full_matrix'
    elif case == 'gaze_training':  # the training cases
        case_dir = base_dir + 'complete_feature_files_gaze_training_cases/'
        out_dir = base_dir + 'feature_matrix_storage_gaze_training_cases/'
        f_params['case_order_rows_file'] = case_dir + 'case_order_rows.txt'
        f_params['target_case_rows_file'] = case_dir + 'gaze_target_case_rows.txt'
        f_params['target_feature_columns_file'] = case_dir + 'gaze_target_feature_columns.txt'
        f_params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
        f_params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
        f_params['feat_targ_samples_outfile'] = out_dir + 'feat_targ_samples_out.txt'
        f_params['item_present_file'] = case_dir + 'items_present-gaze_labeling.txt'
        f_params['feature_matrix_name'] = out_dir + 'full_gaze_training_'
        f_params['target_matrix_name'] = case_dir + 'gaze_target_full_matrix'
    return f_params


def run_feature_selection(params):
    """
    This code runs feature selection for all input experiments.
    """
    def load_samples(file_name):
        samples_dict = {}
        with open(file_name, 'r') as f:
            for full_line in f:
                line = full_line.rstrip()
                if line:
                    if line[0] == '#':  # skip comment lines
                        continue
                    s_line = line.split('\t')
                    samples_dict[s_line[1] + '_' + s_line[3]] = [int(x) for x in s_line[4:]]
        return samples_dict

    mImp_data = np.load(params['mImp_filename'] + '.npy')
    mImp_names = np.load(params['mImp_filename'] + '_names.npy')
    mImp_sets_of_attributes, mImp_names_for_attribute_sets = determine_attribute_sets([str(x) for x in mImp_names.flatten().tolist()])
    rImp_data = np.load(params['rImp_filename'] + '.npy')
    rImp_names = np.load(params['rImp_filename'] + '_names.npy')
    rImp_sets_of_attributes, rImp_names_for_attribute_sets = determine_attribute_sets([str(x) for x in rImp_names.flatten().tolist()])
    print('---Printing data loading numbers---')
    print(mImp_data.shape, mImp_names.shape, len(mImp_sets_of_attributes), len(mImp_names_for_attribute_sets))
    print(rImp_data.shape, rImp_names.shape, len(rImp_sets_of_attributes), len(rImp_names_for_attribute_sets))

    target_samples = load_samples(params['target_samples_outfile'])
    feature_samples = load_samples(params['feature_samples_outfile'])
    target_feature_columns = load_list(params['target_feature_columns_file'])
    target_matrix = np.loadtxt(params['target_matrix_name'] + '.txt', delimiter=',')
    print(len(target_samples), len(feature_samples), len(target_feature_columns), target_matrix.shape)

    model_keys = [x for x in target_samples.keys()]
    target_col_indices = {}
    mImp_out_files_dict = {}
    rImp_out_files_dict = {}
    # ## populate target_col_indices, mImp_out_files_dict, and rImp_out_files_dict
    for key in model_keys:
        if key not in target_col_indices:
            target_col_indices[key] = int(key.split('_')[0])
            mImp_out_files_dict[key] = params['feature_selection_storage'] + key + '-mImp.txt'
            rImp_out_files_dict[key] = params['feature_selection_storage'] + key + '-rImp.txt'
    print('Keys are populated')

    # Needed to run cross fold analysis
    inverse_feature_samples = {}
    inverse_target_samples = {}
    for model_key in model_keys:
        if '_full' in model_key:
            inverse_feature_samples[model_key] = feature_samples[model_key]
            inverse_target_samples[model_key] = target_samples[model_key]
        else:
            inverse_feature_samples[model_key] = []
            inverse_target_samples[model_key] = []
            curr_all_feature_samples = [x for x in feature_samples[model_key.split('_')[0]+'_full']]
            curr_all_target_samples = [x for x in target_samples[model_key.split('_')[0]+'_full']]
            for x in curr_all_feature_samples:
                if x not in feature_samples[model_key]:
                    inverse_feature_samples[model_key].append(x)
            for x in curr_all_target_samples:
                if x not in target_samples[model_key]:
                    inverse_target_samples[model_key].append(x)

    # for current experiment, I only care about the full keys
    full_keys = []
    for model_key in model_keys:
        if '_full' in model_key:
            full_keys.append(model_key)
    model_keys = full_keys
    num_cores = multiprocessing.cpu_count()
    print('='*10 + 'STARTING mIMP' + '='*10)
    print(datetime.datetime.now())
    result = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(staged_feature_inclusion)(mImp_data[inverse_feature_samples[model_key], :], 
                                          target_matrix[inverse_target_samples[model_key], target_col_indices[model_key]], 
                                          mImp_sets_of_attributes, 
                                          params['models_to_use'], 
                                          mImp_out_files_dict[model_key]) for model_key in model_keys)
    print(datetime.datetime.now())
    print('='*10 + 'STARTING rIMP' + '='*10)
    result = Parallel(n_jobs=num_cores, backend='multiprocessing')(
        delayed(staged_feature_inclusion)(rImp_data[inverse_feature_samples[model_key], :], 
                                          target_matrix[inverse_target_samples[model_key], target_col_indices[model_key]], 
                                          rImp_sets_of_attributes, 
                                          params['models_to_use'], 
                                          rImp_out_files_dict[model_key]) for model_key in model_keys)
    print(datetime.datetime.now())
    print('---finished loop---')
    return


@ignore_warnings(category=ConvergenceWarning)
def run_three_model_training(params):
    """
    This code runs cross fold on the training data to select the best model, 
    trains the best model on all of the training data, and then applies the 
    model to the test data. 
    """
    def load_samples(file_name):
        samples_dict = {}
        with open(file_name, 'r') as f:
            for full_line in f:
                line = full_line.rstrip()
                if line:
                    if line[0] == '#':  # skip comment lines
                        continue
                    s_line = line.split('\t')
                    samples_dict[s_line[1] + '_' + s_line[3]] = [int(x) for x in s_line[4:]]
        return samples_dict

    def load_feature_selection_files(prefilename):
        feature_dir = {}
        file_rows = load_list(prefilename+'-mImp.txt')
        for row in file_rows:
            model_type, str_indices = row.split(':')
            if len(str_indices):
                feature_dir[model_type+'-m'] = [int(x) for x in str_indices.split(',')]
            else:
                feature_dir[model_type+'-m'] = [0]
        file_rows = load_list(prefilename+'-rImp.txt')
        for row in file_rows:
            model_type, str_indices = row.split(':')
            if len(str_indices):
                feature_dir[model_type+'-r'] = [int(x) for x in str_indices.split(',')]
            else:
                feature_dir[model_type+'-r'] = [0]
        return feature_dir

    def ave_min_max_of_list(data_list):
        if len(data_list) > 0:
            _result = [round(float(sum(data_list))/len(data_list), 1), min(data_list), max(data_list)]
        else:
            _result = [0, 0, 0]
        return '\t'.join([str(x) for x in _result])

    mImp_data = np.load(params['mImp_filename'] + '.npy')
    mImp_names = np.load(params['mImp_filename'] + '_names.npy')
    rImp_data = np.load(params['rImp_filename'] + '.npy')
    rImp_names = np.load(params['rImp_filename'] + '_names.npy')

    mImp_test_data = np.load(params['mImp_test_filename'] + '.npy')
    rImp_test_data = np.load(params['rImp_test_filename'] + '.npy')

    target_samples = load_samples(params['target_samples_outfile'])
    feature_samples = load_samples(params['feature_samples_outfile'])
    target_feature_columns = load_list(params['target_feature_columns_file'])
    target_matrix = np.loadtxt(params['target_matrix_name'] + '.txt', delimiter=',')
    model_keys = target_samples.keys()
    full_keys = []
    for key in model_keys:
        if '_full' in key:
            full_keys.append(key)

    target_col_indices = {}
    # ## populate target_col_indices
    for key in model_keys:
        if key not in target_col_indices:
            target_col_indices[key] = int(key.split('_')[0])

    clf_lr = LogisticRegression(penalty='l2', random_state=42, solver='liblinear', max_iter=200)
    clf_sv = SVC(C=1, gamma='scale', probability=True, random_state=42)
    clf_rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight={0: 1.00, 1: 1.00})

    model_types_out = open(params['result_out_dir'] + 'model_types.txt', 'w')
    predictions_out = open(params['result_out_dir'] + 'predictions_out.txt', 'w')
    probabilities_out = open(params['result_out_dir'] + 'probabilities_out.txt', 'w')
    model_types_out.write('#model_id,model_type,max_train_auroc\n')

    # Sanity checks
    print('---Sanity checks---')
    print(mImp_data.shape, mImp_names.shape, mImp_test_data.shape, 'mImp: x_train, x_names, x_test')
    print(rImp_data.shape, rImp_names.shape, rImp_test_data.shape, 'rImp: x_train, x_names, x_test')
    print(datetime.datetime.now())

    feature_selection_size_dict = {'lr-m': [], 'sv-m': [], 'rf-m': [], 'lr-r': [], 'sv-r': [], 'rf-r': []}

    for key in full_keys:
        model_name = target_feature_columns[int(key.rstrip('_full'))]
        print (key, model_name)
        training_y = target_matrix[target_samples[key], target_col_indices[key]]

        # load feature selections
        curr_m_data = mImp_data[feature_samples[key], :]
        curr_r_data = rImp_data[feature_samples[key], :]
        feature_selection_dict = load_feature_selection_files(params['feature_selection_storage']+key)
        num_folds = min(np.count_nonzero(training_y), training_y.shape[0]-np.count_nonzero(training_y))

        if num_folds < 3:
            continue

        # cross validate models
        lr_m_scores = cross_val_score(clf_lr, curr_m_data[:, feature_selection_dict['lr-m']], training_y, cv=num_folds)
        sv_m_scores = cross_val_score(clf_sv, curr_m_data[:, feature_selection_dict['sv-m']], training_y, cv=num_folds)
        rf_m_scores = cross_val_score(clf_rf, curr_m_data[:, feature_selection_dict['rf-m']], training_y, cv=num_folds)
        lr_r_scores = cross_val_score(clf_lr, curr_r_data[:, feature_selection_dict['lr-r']], training_y, cv=num_folds)
        sv_r_scores = cross_val_score(clf_sv, curr_r_data[:, feature_selection_dict['sv-r']], training_y, cv=num_folds)
        rf_r_scores = cross_val_score(clf_rf, curr_r_data[:, feature_selection_dict['rf-r']], training_y, cv=num_folds)

        # find best cross validated model
        scores = [lr_m_scores.mean(), sv_m_scores.mean(), rf_m_scores.mean(), lr_r_scores.mean(), sv_r_scores.mean(), rf_r_scores.mean()]
        max_score = max(scores)
        best_score = scores.index(max_score)

        if best_score == 0:
            model_type = 'lr-m'
            clf_lr.fit(curr_m_data[:, feature_selection_dict['lr-m']], training_y) 
            train_prediction_probabilities = clf_lr.predict_proba(curr_m_data[:, feature_selection_dict['lr-m']])[:, 1]
            predictions = clf_lr.predict(mImp_test_data[:, feature_selection_dict['lr-m']])
            prediction_probabilities = clf_lr.predict_proba(mImp_test_data[:, feature_selection_dict['lr-m']])[:, 1]
            feature_selection_size_dict['lr-m'].append(len(feature_selection_dict['lr-m']))
        elif best_score == 1: 
            model_type = 'sv-m'
            clf_sv.fit(curr_m_data[:, feature_selection_dict['sv-m']], training_y)
            train_prediction_probabilities = clf_sv.predict_proba(curr_m_data[:, feature_selection_dict['sv-m']])[:, 1]
            predictions = clf_sv.predict(mImp_test_data[:, feature_selection_dict['sv-m']])
            prediction_probabilities = clf_sv.predict_proba(mImp_test_data[:, feature_selection_dict['sv-m']])[:, 1]
            feature_selection_size_dict['sv-m'].append(len(feature_selection_dict['sv-m']))
        elif best_score == 2: 
            model_type = 'rf-m'
            clf_rf.fit(curr_m_data[:, feature_selection_dict['rf-m']], training_y)
            train_prediction_probabilities = clf_rf.predict_proba(curr_m_data[:, feature_selection_dict['rf-m']])[:, 1]
            predictions = clf_rf.predict(mImp_test_data[:, feature_selection_dict['rf-m']])
            prediction_probabilities = clf_rf.predict_proba(mImp_test_data[:, feature_selection_dict['rf-m']])[:, 1]
            feature_selection_size_dict['rf-m'].append(len(feature_selection_dict['rf-m']))
        elif best_score == 3: 
            model_type = 'lr-r'
            clf_lr.fit(curr_r_data[:, feature_selection_dict['lr-r']], training_y) 
            train_prediction_probabilities = clf_lr.predict_proba(curr_r_data[:, feature_selection_dict['lr-r']])[:, 1]
            predictions = clf_lr.predict(rImp_test_data[:, feature_selection_dict['lr-r']])
            prediction_probabilities = clf_lr.predict_proba(rImp_test_data[:, feature_selection_dict['lr-r']])[:, 1]
            feature_selection_size_dict['lr-r'].append(len(feature_selection_dict['lr-r']))
        elif best_score == 4: 
            model_type = 'sv-r'
            clf_sv.fit(curr_r_data[:, feature_selection_dict['sv-r']], training_y)
            train_prediction_probabilities =  clf_sv.predict_proba(curr_r_data[:, feature_selection_dict['sv-r']])[:, 1]
            predictions = clf_sv.predict(rImp_test_data[:, feature_selection_dict['sv-r']])
            prediction_probabilities = clf_sv.predict_proba(rImp_test_data[:, feature_selection_dict['sv-r']])[:, 1]
            feature_selection_size_dict['sv-r'].append(len(feature_selection_dict['sv-r']))
        elif best_score == 5: 
            model_type = 'rf-r'
            clf_rf.fit(curr_r_data[:, feature_selection_dict['rf-r']], training_y) 
            train_prediction_probabilities = clf_rf.predict_proba(curr_r_data[:, feature_selection_dict['rf-r']])[:, 1]
            predictions = clf_rf.predict(rImp_test_data[:, feature_selection_dict['rf-r']])
            prediction_probabilities = clf_rf.predict_proba(rImp_test_data[:, feature_selection_dict['rf-r']])[:, 1]
            feature_selection_size_dict['rf-r'].append(len(feature_selection_dict['rf-r']))
        else:
            print('error for ' + key)
            model_types, predictions, prediction_probabilities = ['ERROR', 'ERROR', 'ERROR']

        model_types_out.write(key.rstrip('_full') + ',' + model_type + ',' + str(max_score) + '\n')
        predictions_out.write(model_name + ',' + ','.join([str(x) for x in predictions]) + '\n')
        probabilities_out.write(key.rstrip('_full') + ',' + ','.join([str(x) for x in prediction_probabilities]) + '\n')

    model_types_out.close()
    predictions_out.close()
    probabilities_out.close()

    # print feature selection and model details
    fssd = feature_selection_size_dict
    print('model', 'count_used', 'average_features', 'min_features', 'max_features')
    print('lr-m', len(fssd['lr-m']), ave_min_max_of_list(fssd['lr-m']))
    print('sv-m', len(fssd['sv-m']), ave_min_max_of_list(fssd['sv-m']))
    print('rf-m', len(fssd['rf-m']), ave_min_max_of_list(fssd['rf-m']))
    print('lr-r', len(fssd['lr-r']), ave_min_max_of_list(fssd['lr-r']))
    print('sv-r', len(fssd['sv-r']), ave_min_max_of_list(fssd['sv-r']))
    print('rf-r', len(fssd['rf-r']), ave_min_max_of_list(fssd['rf-r']))

    print('Complete')
    print(datetime.datetime.now())

    return


def populate_feature_selection_params(type_idx=0):
    f_params = {}
    cases = ['manual_training', 'gaze_training']
    case = cases[type_idx]
    base_dir = '../'
    if case == 'manual_training':  # manually labeled trainig
        case_dir = base_dir + 'complete_feature_files_manual_training_cases/'
        out_dir = base_dir + 'feature_matrix_storage_manual_training_cases/'
        f_params['feature_selection_storage'] = out_dir + 'feature_selection_storage/'
        f_params['mImp_filename'] = out_dir + 'full_manual_training_mImp'
        f_params['rImp_filename'] = out_dir + 'full_manual_training_rImp'
        f_params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
        f_params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
        f_params['target_feature_columns_file'] = case_dir + 'manual_target_feature_columns.txt'
        f_params['target_matrix_name'] = case_dir + 'manual_target_full_matrix'
        f_params['models_to_use'] = ['lr', 'sv', 'rf']
        f_params['mImp_test_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation_mImp'
        f_params['rImp_test_filename'] = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation_rImp'
        f_params['result_out_dir'] = base_dir + 'evaluation_study_models_manual/'
    elif case == 'gaze_training':  # eye labeled training
        case_dir = base_dir + 'complete_feature_files_gaze_training_cases/'
        out_dir = base_dir + 'feature_matrix_storage_gaze_training_cases/'
        f_params['feature_selection_storage'] = out_dir + 'feature_selection_storage/'
        f_params['mImp_filename'] = out_dir + 'full_gaze_training_mImp'
        f_params['rImp_filename'] = out_dir + 'full_gaze_training_rImp'
        f_params['feature_samples_outfile'] = out_dir + 'feature_samples_out.txt'
        f_params['target_samples_outfile'] = out_dir + 'target_samples_out.txt'
        f_params['target_feature_columns_file'] = case_dir + 'gaze_target_feature_columns.txt'
        f_params['target_matrix_name'] = case_dir + 'gaze_target_full_matrix'
        f_params['models_to_use'] = ['lr', 'sv', 'rf']
        f_params['mImp_test_filename'] = base_dir + 'feature_matrix_storage_gaze_eval_cases/full_gaze_eval_mImp'
        f_params['rImp_test_filename'] = base_dir + 'feature_matrix_storage_gaze_eval_cases/full_gaze_eval_rImp'
        f_params['result_out_dir'] = base_dir + 'evaluation_study_models_gaze/'

    return f_params


if __name__ == "__main__":

    # clean columns and impute data
    if False:
        params = populate_imputation_params(type_idx=0)
        imputation(params)
        params = populate_imputation_params(type_idx=1)
        imputation(params)

    # ## select target columns and sample rows
    if False:
        params = populate_sample_rows_params(type_idx=0)
        determine_feature_matrix_and_target_matrix_rows(params)
        params = populate_sample_rows_params(type_idx=1)
        determine_feature_matrix_and_target_matrix_rows(params)

    # ## run feature selection
    if False:
        params = populate_feature_selection_params(type_idx=0)
        run_feature_selection(params)
        params = populate_feature_selection_params(type_idx=1)
        run_feature_selection(params)

    # ## run models folds
    if False:
        params = populate_feature_selection_params(type_idx=0)
        run_three_model_training(params)
        params = populate_feature_selection_params(type_idx=1)
        run_three_model_training(params)


"""
--------info------

> Last run on 2019/09/29 
<<<first if>>>
Train data loaded:  (134, 51839) (51839, 1)
Eval data loaded:  (18, 51839) (51839, 1)
====================
Train columns cleaned:  (134, 13596) (13596, 1)
Eval columns cleaned:  (18, 13596) (13596, 1)
====================
0        2019-09-29 20:31:51.207777
500      2019-09-29 20:32:32.731632
...
13500    2019-09-29 20:47:59.574343
Train rImp:  (134, 10254) (10254, 1) True
Eval rImp:  (18, 10254) (10254, 1) True
====================
Train mImp:  (134, 13596) (13596, 1) True
Eval mImp:  (18, 13596) (13596, 1) True
++++++++++++++++++++
0        2019-09-29 21:05:51.751228
500      2019-09-29 21:06:32.999068
...
13500    2019-09-29 21:21:48.289250
Train rImp:  (134, 10254) (10254, 1) True
Eval rImp:  (18, 10254) (10254, 1) True
====================
Train mImp:  (134, 13596) (13596, 1) True
Eval mImp:  (18, 13596) (13596, 1) True
++++++++++++++++++++

<<<second if>>>
(no output is printed for this)

<<<third if>>>
---Printing data loading numbers---
(134, 13596) (13596, 1) 1277 1277
(134, 10254) (10254, 1) 1164 1164
528 528 88 (134, 88)
Keys are populated
==========STARTING mIMP==========
2019-09-29 10:36:00.000000
2019-09-29 11:17:00.000000
==========STARTING rIMP==========
2019-09-30 00:06:34.365004
---finished loop---
---Printing data loading numbers---
(134, 13596) (13596, 1) 1277 1277
(134, 10254) (10254, 1) 1164 1164
690 690 115 (134, 115)
Keys are populated
==========STARTING mIMP==========
2019-09-30 00:06:44.240883
2019-09-30 00:55:11.860182
==========STARTING rIMP==========
2019-09-30 01:59:18.320298
---finished loop---

<<<forth if>>>
---Sanity checks---
(134, 13596) (13596, 1) (18, 13596) mImp: x_train, x_names, x_test
(134, 10254) (10254, 1) (18, 10254) rImp: x_train, x_names, x_test
2019-09-30 08:14:11.720144
0_full NEUTR
1_full senna
...
87_full ICA
model count_used average_features min_features max_features
lr-m 18 207.4   52      562
sv-m 18 2366.3  958     4391
rf-m 27 336.0   86      935
lr-r 10 659.2   57      1275
sv-r 0 0        0       0
rf-r 14 806.1   187     1832
Complete
2019-09-30 08:27:53.354030
---Sanity checks---
(134, 13596) (13596, 1) (18, 13596) mImp: x_train, x_names, x_test
(134, 10254) (10254, 1) (18, 10254) rImp: x_train, x_names, x_test
2019-09-30 07:10:50.348076
0_full NEUTR
1_full senna
...
114_full TPROT
model count_used average_features min_features max_features
lr-m 24 176.3   28      447
sv-m 9 2108.9   1663    2723
rf-m 37 341.9   57      1021
lr-r 15 304.9   61      1047
sv-r 0 0        0       0
rf-r 30 613.8   65      2663
Complete
2019-09-30 07:32:54.486934
-------------------
"""
