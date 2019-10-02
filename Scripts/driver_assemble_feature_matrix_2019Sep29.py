"""
driver_assemble_feature_matrix_2019Sep29.py
version 2019Sep29
package https://github.com/ajk77/jmir-2019
Created by AndrewJKing.com|@andrewsjourney

Code for building final version of assemble_feature_matrix
For more details see https://github.com/ajk77/PatientPy

---DEPENDENCIES---
https://github.com/ajk77/PatientPy

Must first run PatientPy/patient_pickler.py to store data structures that are used here.
^ set pkl_dir to the same value as was used in patient_pickler.
Must second run PatientPy/create_feature_vectors.py once for each directory filled by create_feature_vectors.py.
Then run this file.

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

import sys
sys.path.insert(0, '../../../../git_projects/patientpy')
from assemble_feature_matrix import assemble_feature_matrix


cases = ['manual_training', 'gaze_training', 'evaluation', 'gaze_eval']
case = cases[3]
base_dir = '../'
if case == 'manual_training':  # the first four 'burn in' cases
    feature_dir = base_dir + 'complete_feature_files_manual_training_cases/'
    output_filename = base_dir + 'feature_matrix_storage_manual_training_cases/full_manual_training'
    add_feat = ['diagnosis_features']
elif case == 'gaze_training':  # the training cases
    feature_dir = base_dir + 'complete_feature_files_gaze_training_cases/'
    output_filename = base_dir + 'feature_matrix_storage_gaze_training_cases/full_gaze_training'
    add_feat = ['diagnosis_features']
elif case == 'evaluation':  # the evaluation cases
    feature_dir = base_dir + 'complete_feature_files_evaluation_cases/'
    output_filename = base_dir + 'feature_matrix_storage_evaluation_cases/full_evaluation'
    add_feat = ['diagnosis_features']
elif case == 'gaze_eval':  # the evaluation cases
    feature_dir = base_dir + 'complete_feature_files_evaluation_cases/'
    output_filename = base_dir + 'feature_matrix_storage_gaze_eval_cases/full_gaze_eval'
    add_feat = ['diagnosis_features']

feat_types = ['demo', 'io', 'med', 'micro', 'procedure', 'root']

column_match = []  # 

assemble_feature_matrix(feature_dir, output_filename, feature_types_to_include=feat_types, additional_features=add_feat)

"""
---------info---------------
>>run where case=cases[0] on 2019/09/29
======demo======
loaded:  (134, 6) (6,)
concatenated:  (134, 6) (6,)
======io======
loaded:  (134, 14) (14,)
concatenated:  (134, 20) (20,)
======med======
loaded:  (134, 10863) (10863,)
concatenated:  (134, 10883) (10883,)
======micro======
loaded:  (134, 40) (40,)
concatenated:  (134, 10923) (10923,)
======procedure======
loaded:  (134, 1576) (1576,)
concatenated:  (134, 12499) (12499,)
======root======
loaded:  (134, 39339) (39339,)
concatenated:  (134, 51838) (51838,)
------diagnosis_features------
(134, 1) (1,)
concatenated:  (134, 51839) (51839,)
additional featueres added:  (134, 51839) (51839,)
full:  (134, 51839) (51839,)

>>run where case=cases[1] on 2019/09/29
======demo======
loaded:  (134, 6) (6,)
concatenated:  (134, 6) (6,)
======io======
loaded:  (134, 14) (14,)
concatenated:  (134, 20) (20,)
======med======
loaded:  (134, 10863) (10863,)
concatenated:  (134, 10883) (10883,)
======micro======
loaded:  (134, 40) (40,)
concatenated:  (134, 10923) (10923,)
======procedure======
loaded:  (134, 1576) (1576,)
concatenated:  (134, 12499) (12499,)
======root======
loaded:  (134, 39339) (39339,)
concatenated:  (134, 51838) (51838,)
------diagnosis_features------
(134, 1) (1,)
concatenated:  (134, 51839) (51839,)
additional featueres added:  (134, 51839) (51839,)
full:  (134, 51839) (51839,)

>>run where case=cases[2] on 2019/09/29
======demo======
loaded:  (18, 6) (6,)
concatenated:  (18, 6) (6,)
======io======
loaded:  (18, 14) (14,)
concatenated:  (18, 20) (20,)
======med======
loaded:  (18, 10863) (10863,)
concatenated:  (18, 10883) (10883,)
======micro======
loaded:  (18, 40) (40,)
concatenated:  (18, 10923) (10923,)
======procedure======
loaded:  (18, 1576) (1576,)
concatenated:  (18, 12499) (12499,)
======root======
loaded:  (18, 39339) (39339,)
concatenated:  (18, 51838) (51838,)
------diagnosis_features------
(18, 1) (1,)
concatenated:  (18, 51839) (51839,)
additional features added:  (18, 51839) (51839,)
full:  (18, 51839) (51839,)

>>run where case=cases[3] on 2019/09/29
======demo======
loaded:  (18, 6) (6,)
concatenated:  (18, 6) (6,)
======io======
loaded:  (18, 14) (14,)
concatenated:  (18, 20) (20,)
======med======
loaded:  (18, 10863) (10863,)
concatenated:  (18, 10883) (10883,)
======micro======
loaded:  (18, 40) (40,)
concatenated:  (18, 10923) (10923,)
======procedure======
loaded:  (18, 1576) (1576,)
concatenated:  (18, 12499) (12499,)
======root======
loaded:  (18, 39339) (39339,)
concatenated:  (18, 51838) (51838,)
------diagnosis_features------
(18, 1) (1,)
concatenated:  (18, 51839) (51839,)
additional featueres added:  (18, 51839) (51839,)
full:  (18, 51839) (51839,)
--------------------------
"""
