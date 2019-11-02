# jmir-2019

This package is for reproducing the experiments conducted in the JMIR-submitted paper "Harnessing Eye Tracking to 
Selectively Highlight Medical Record Information"

## Citing this work

Please cite:<br>
King AJ, Cooper GF, Clermont G, Hochheiser H, Hauskrecht M, Sittig DF, Visweswaran S. Harnessing Eye Tracking to 
Selectively Highlight Medical Record Information. [under review]

Also see:<br>
King AJ, Hochheiser H, Visweswaran S, Clermont G, Cooper GF. Eye-tracking for clinical decision support: A method to capture automatically what physicians are viewing in the EMR. AMIA Joint Summits. 2017 Mar 27-30; San Francisco, California p 512-521. (https://www.ncbi.nlm.nih.gov/pubmed/28815151)


## Description

This directory is for reproducing the results of the machine learning paper, Harnessing Eye Tracking to Selectively Highlight Medical Record Data.  



### Prerequisites

The code was run using Python 3.7.3. See jmir-2019/freeze.txt for full environmental setup. 

If reproducing the analysis:
* Prerequisites for running jmir-2019/Scripts/overall_model_performance_2019Oct01.py 
    * See jmir-2019/freeze.txt for full environmental setup

If you want to reproduce the full experiment:
* Please contact the authors to discuss access to a transormed version of the de-identified patient data they used.

* Prerequisites for running jmir-2019/Scripts/driver_assemble_feature_matrix_2019sep29.py
    * First view and download download PatientPy (https://github.com/ajk77/PatientPy)
    * The data stored in the jmir-2019/complete_feature_files_* folders is the output of PatientPy/create_feature_vectors.py
    * Running jmir-2019/Scripts/driver_assemble_feature_matrix_2019sep29.py applies PatientPy/assemble_feature_matrix.py

* Prerequisites for running jmir-2019/Scripts/instantiate_experiment_2019sep09.py
    * First view and download download RegressiveImputer (https://github.com/ajk77/RegressiveImputer)
    * First view and download download PatientPyFeatureSelection (https://github.com/ajk77/PatientPyFeatureSelection)

### Installing

1. Download jmir-2019
2. Create a Python 3.7 virtual environment
3. Install the requirements listed in jmir-2019/freeze.txt<br>

## Deployment

* If reproducing the analysis:  
    * In the __main__ definition for jmir-2019/Scripts/overall_model_performance_2019Oct01.py, change the two conditionals to be True. (False by default to prevent accidental re-running of results).

* If you want to reproduce the full experiment:
    * Please see the prerequisites section. 
    * jmir-2019/Scripts/driver_assemble_feature_matrix_2019sep29.py should be run four times. In each run, change the index of "case = cases[]". The output populates the feature_matrix_stroage_* directories. 
    * In the __main__ definition for jmir-2019/Scripts/instantiate_experiment_2019sep09.py, change the four conditional's to be True. (These are false by default so that you do not need to run all of the code at once. This code takes 12 hours on an Intel(R) Core(TM) i7-870 CPU).
    * In the __main__ definition for jmir-2019/Scripts/overall_model_performance_2019Oct01.py, change the two conditional's to be True. (False by default to prevent accidental re-running of results).


The results are printed in evaluation_study_models_gaze and evaluation_study_models_manual directories. 

### Note
In the PatientPy package, the same extraction code is used for all laboratory tests, vital signs, and ventilator settings. This results in the 
extraction of 47 features per variable; however, the features that are not applicable for a variable are set to null 
and dropped before imputation, thus giving the proper dimensions for each variable type. 

If you find a possible error in any of this code or documentation, please inform Andrew J King. We appreciate your contribution to this science. 

## Versioning

Version 2019Oct01 For the versions available, see https://github.com/ajk77/jmir-2019

## Authors

* Andrew J King - Doctoral Candidate (at time of creation)
	* Website (https://www.andrewjking.com/)
	* Twitter (https://twitter.com/andrewsjourney)
* Shyam Visweswaran - Principal Investigator
	* Website (http://www.thevislab.com/)
	* Twitter (https://twitter.com/Shyam_Vis)
* Gregory F Cooper - Doctoral Advisor

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2019Oct01 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Acknowledgments

* Harry Hochheiser
	* Twitter (https://twitter.com/hshoch)
* Gilles Clermont
* Milos Hauskrecht 
