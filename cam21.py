#!/usr/bin/env python
# coding: utf-8

# Complete Redesign of the Conceputal Alignment Method (CAM)
# 
# Data Set: Hospital Inpatient Discharges (SPARCS De-Identified): 2017
# URL: https://health.data.ny.gov/dataset/Hospital-Inpatient-Discharges-SPARCS-De-Identified/22g3-z7e7
#
# Last revision: September 15, 2023
# Autor: Wolfgang Maass
# Closest publication: AMCIS submission 2023
# Version: 2.2
#
# Current version: adaptation to Python 3.9.13
# Date: 2023-12-20

import io
import base64  # Import base64 module
import matplotlib.pyplot as plt

import oolib
from oolib import *

import calign21
from calign21 import * # that's the problem for reloading issues!!! use calign21.function instead
import os
import shutil
import tabulate

import importlib
importlib.reload(calign21)

#import tensorflow as tf
import GPUtil
import numba

import cProfile
import pstats

import logging
logger = logging.getLogger(__name__)

import functools
import time



# domain and dataset selection
app_names = ["HIS17-two-classes"]  
#app_names = ["HIS17-random"] # ["HIS17", "HIS17-2", "HIS17-two-classes"]  
#app_names = ['LOAN'] #, 'LOAN-2']

benchmark_model = '' # 'HIS17-2'

# number of overall iterations of the algorithm
main_n = 1
scale = 3
epsilon = 0.001
eps_threshold = 0.1
concept_max = 2

# test GPU
GPUs = GPUtil.getGPUs()
print(GPUs)

model_path = './models/' + app_names[0] + '/model'

# if directory './models/' + app_names[0] does not exist, create
if not os.path.exists('./models/' + app_names[0]):
    os.makedirs('./models/' + app_names[0])
    print("Created folder: ", './models/' + app_names[0])

if GPUs:
    path_data = '../../data/'
else:
    path_data = './data/'
    #path_data = './CAData/'

# Define seed for repeatability
SEED = 1
np.random.seed(SEED)

# threshold
t = 0.001

# Simulation factor
sim = 1

# Iteration index
i_id = 0

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

verbose = 0
details = 0

prediction_mode = 1 # 0: all concepts are predicted, 1: only the decision concept is predicted

### DATA

#res_path = './results/'

#cleanup_files(res_path)

def main():
    ### CORE PROCESS OF MEM

    # make attribute_iterations a global variable and initialize as empty dataframe
    #global attribute_iterations

    global cd
    global mi
    global ci
    global cut
    #global cluster_cut
    global r2_threshold
    global epsilon
    global eps_threshold
    global prediction_mode
    global concept_max

    global path_results    
    global path_figures
    global path_data
    global main_n
    global scale

    global features_randomize

    #concept_max = 0 # maximum number of concepts to be created
    #diff_sep_features = 0 # number of different features for sep = 1

    for app_name in app_names:

        print("-----------------------------------")
        print("App name: ", app_name)
        print("-----------------------------------\n")

        path_results = './results/'+ app_name + '/'
        path_figures = './figures/'+ app_name + '/'
        path_graphs = './graphs/'+ app_name + '/'

        features_randomize = ['Length of Stay']
        features_randomize = []

        # Configure logging to save to a file
        #logging.basicConfig(filename=path_results + 'function_logs.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        delete_file_if_exists(path_results, "r2_t_hist.csv")
        delete_file_if_exists(path_results, "r2.csv")

        # loop n times function f
        for mi in range(0, main_n):

            if app_name == "HC-benchmark":

                scale = 3 # size of the dataset fraction
                #step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index

                #d_name = "onehot_prep.csv"
                #d_name_short = "HIS17"

                path_data = '~/Dropbox/CAData/'
                d_name = "hdma_1M.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.7, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['hospital'] = {
                    'Hospital County_Bronx', 
                    'Hospital County_Manhattan',
                    'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 
                    'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 
                    'Facility Name_Mount Sinai Beth Israel', 
                    'Facility Name_Mount Sinai Hospital', 
                    'Facility Name_St. Marys Healthcare', 
                    'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 
                    'Facility Name_Staten Island University Hosp-North', 
                    'Facility Name_Staten Island University Hosp-South', 
                    'Facility Name_Strong Memorial Hospital', 
                    'Facility Name_Syosset Hospital', 
                    'Facility Name_TLC Health Network Lake Shore Hospital', 
                    'Facility Name_The Burdett Care Center', 
                    'Facility Name_The Unity Hospital of Rochester', 
                    'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 
                    'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 
                    'Facility Name_UPMC Chautauqua at WCA', 
                    'Facility Name_UPSTATE University Hospital at Community General', 
                    'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 
                    'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 
                    'Facility Name_United Memorial Medical Center North Street Campus', 
                    'Facility Name_University Hospital', 
                    'Facility Name_University Hospital - Stony Brook Southampton Hospital', 
                    'Facility Name_University Hospital SUNY Health Science Center', 
                    'Facility Name_University Hospital of Brooklyn', 
                    'Facility Name_Vassar Brothers Medical Center', 
                    'Facility Name_Westchester Medical Center', 
                    'Facility Name_Westfield Memorial Hospital Inc', 
                    'Facility Name_White Plains Hospital Center', 
                    'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 
                    'Facility Name_Women And Childrens Hospital Of Buffalo', 
                    'Facility Name_Woodhull Medical & Mental Health Center', 
                    'Facility Name_Wyckoff Heights Medical Center', 
                    'Facility Name_Wyoming County Community Hospital', 
                    'Emergency Department Indicator_N', 
                    'Emergency Department Indicator_Y' 
                    #'Permanent Facility Id'
                    }

                cd['patient'] = {
                    'Gender_F', 
                    'Gender_M',
                    'Payment Typology 3_Self-Pay',
                    'Race_Black/African American', 
                    'Race_Other Race', 
                    'Race_White',
                    'Payment Typology 1_Medicaid', 
                    'Payment Typology 1_Medicare', 
                    'Payment Typology 1_Private Health Insurance', 
                    'Zip Code_100', 
                    'Zip Code_104', 
                    'Zip Code_112', 
                    'Payment Typology 2_Medicaid', 
                    'Payment Typology 2_Self-Pay', 
                    'Ethnicity_Not Span/Hispanic', 
                    'Ethnicity_Spanish/Hispanic',  
                    'Age Group_0 to 17', 
                    'Age Group_18 to 29', 
                    'Age Group_30 to 49', 
                    'Age Group_50 to 69', 
                    'Age Group_70 or Older', 
                    'Patient Disposition_Home or Self Care', 
                    'Patient Disposition_Home w/ Home Health Services', 
                    }
                
                cd['admission'] = {
                    'Length of Stay', 
                    'Type of Admission_Elective', 
                    'Type of Admission_Emergency',
                    } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

                cd['illness'] = { 
                    'APR MDC Code_4', 
                    'APR MDC Code_5', 
                    'APR MDC Code_14', 
                    'APR Risk of Mortality_Major', 
                    'APR Risk of Mortality_Minor', 
                    'APR Risk of Mortality_Moderate', 
                    'CCS Procedure Code_0', 
                    'CCS Procedure Code_231', 
                    'APR Severity of Illness Code_1', 
                    'APR Severity of Illness Code_2', 
                    'APR Severity of Illness Code_3', 
                    'APR Medical Surgical Description_Medical', 
                    'APR Medical Surgical Description_Not Applicable', 
                    'APR Medical Surgical Description_Surgical',
                    }

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'hospital': cd['hospital'], 
                    'patient': cd['patient'], 
                    'admission': cd['admission'], 
                    'illness': cd['illness']
                }

                predictors_all = []
                predictors_all.extend(cd['hospital'])
                predictors_all.extend(cd['patient'])
                predictors_all.extend(cd['admission'])
                predictors_all.extend(cd['illness'])
                predictors_all.extend(cd['decision'])

                # f_dict = {
                # "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                # "hospital": [cd['patient'], cd['admission'], cd['illness']],
                # "patient": [cd['hospital'], cd['admission'], cd['illness']],
                # "admission": [cd['hospital'], cd['patient'], cd['illness']],
                # "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                f_dict = {
                    "decision": [cd['hospital'], cd['patient'],     cd['admission'], cd['illness']],
                    "hospital": [cd['patient'],  cd['admission'],   cd['illness']],
                    "patient":  [cd['hospital'], cd['admission'],   cd['illness']],
                    "admission":[cd['hospital'], cd['patient'],     cd['illness']],
                    "illness":  [cd['hospital'], cd['patient'],     cd['admission']]}
                
                c_dict = {
                    "decision": [["hospital", "patient", "admission", "illness"], "decision", 'Total Costs', 'regression'],
                    "hospital": [["patient", "admission", "illness"], "hospital", 'Emergency Department Indicator_N', 'regression'],
                    "patient": [["hospital", "admission", "illness"],"patient", 'Age Group_70 or Older', 'regression'],
                    "admission": [["hospital", "patient", "illness"], "admission", 'Length of Stay', 'regression'],
                    "illness": [["hospital", "patient", "admission"],"illness", 'APR Risk of Mortality_Major', 'regression']
                }

            if app_name == "HC-four-classes":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 2 # maximum number of concepts to be created

                features_randomize = ['Length of Stay']

                d_name = "onehot_prep.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.7, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['hospital'] = {
                    'Hospital County_Bronx', 
                    'Hospital County_Manhattan',
                    'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 
                    'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 
                    'Facility Name_Mount Sinai Beth Israel', 
                    'Facility Name_Mount Sinai Hospital', 
                    'Facility Name_St. Marys Healthcare', 
                    'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 
                    'Facility Name_Staten Island University Hosp-North', 
                    'Facility Name_Staten Island University Hosp-South', 
                    'Facility Name_Strong Memorial Hospital', 
                    'Facility Name_Syosset Hospital', 
                    'Facility Name_TLC Health Network Lake Shore Hospital', 
                    'Facility Name_The Burdett Care Center', 
                    'Facility Name_The Unity Hospital of Rochester', 
                    'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 
                    'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 
                    'Facility Name_UPMC Chautauqua at WCA', 
                    'Facility Name_UPSTATE University Hospital at Community General', 
                    'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 
                    'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 
                    'Facility Name_United Memorial Medical Center North Street Campus', 
                    'Facility Name_University Hospital', 
                    'Facility Name_University Hospital - Stony Brook Southampton Hospital', 
                    'Facility Name_University Hospital SUNY Health Science Center', 
                    'Facility Name_University Hospital of Brooklyn', 
                    'Facility Name_Vassar Brothers Medical Center', 
                    'Facility Name_Westchester Medical Center', 
                    'Facility Name_Westfield Memorial Hospital Inc', 
                    'Facility Name_White Plains Hospital Center', 
                    'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 
                    'Facility Name_Women And Childrens Hospital Of Buffalo', 
                    'Facility Name_Woodhull Medical & Mental Health Center', 
                    'Facility Name_Wyckoff Heights Medical Center', 
                    'Facility Name_Wyoming County Community Hospital', 
                    'Emergency Department Indicator_N', 
                    'Emergency Department Indicator_Y' 
                    #'Permanent Facility Id'
                    }

                cd['patient'] = {
                    'Gender_F', 
                    'Gender_M',
                    'Payment Typology 3_Self-Pay',
                    'Race_Black/African American', 
                    'Race_Other Race', 
                    'Race_White',
                    'Payment Typology 1_Medicaid', 
                    'Payment Typology 1_Medicare', 
                    'Payment Typology 1_Private Health Insurance', 
                    'Zip Code_100', 
                    'Zip Code_104', 
                    'Zip Code_112', 
                    'Payment Typology 2_Medicaid', 
                    'Payment Typology 2_Self-Pay', 
                    'Ethnicity_Not Span/Hispanic', 
                    'Ethnicity_Spanish/Hispanic',  
                    'Age Group_0 to 17', 
                    'Age Group_18 to 29', 
                    'Age Group_30 to 49', 
                    'Age Group_50 to 69', 
                    'Age Group_70 or Older', 
                    'Patient Disposition_Home or Self Care', 
                    'Patient Disposition_Home w/ Home Health Services', 
                    }
                
                cd['admission'] = {
                    'Length of Stay', 
                    'Type of Admission_Elective', 
                    'Type of Admission_Emergency',
                    } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

                cd['illness'] = { 
                    'APR MDC Code_4', 
                    'APR MDC Code_5', 
                    'APR MDC Code_14', 
                    'APR Risk of Mortality_Major', 
                    'APR Risk of Mortality_Minor', 
                    'APR Risk of Mortality_Moderate', 
                    'CCS Procedure Code_0', 
                    'CCS Procedure Code_231', 
                    'APR Severity of Illness Code_1', 
                    'APR Severity of Illness Code_2', 
                    'APR Severity of Illness Code_3', 
                    'APR Medical Surgical Description_Medical', 
                    'APR Medical Surgical Description_Not Applicable', 
                    'APR Medical Surgical Description_Surgical',
                    }

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'hospital': cd['hospital'], 
                    'patient': cd['patient'], 
                    'admission': cd['admission'], 
                    'illness': cd['illness']
                }

                predictors_all = []
                predictors_all.extend(cd['hospital'])
                predictors_all.extend(cd['patient'])
                predictors_all.extend(cd['admission'])
                predictors_all.extend(cd['illness'])
                predictors_all.extend(cd['decision'])

                # f_dict = {
                # "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                # "hospital": [cd['patient'], cd['admission'], cd['illness']],
                # "patient": [cd['hospital'], cd['admission'], cd['illness']],
                # "admission": [cd['hospital'], cd['patient'], cd['illness']],
                # "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                f_dict = {
                    "decision": [cd['hospital'], cd['patient'],     cd['admission'], cd['illness']],
                    "hospital": [cd['patient'],  cd['admission'],   cd['illness']],
                    "patient":  [cd['hospital'], cd['admission'],   cd['illness']],
                    "admission":[cd['hospital'], cd['patient'],     cd['illness']],
                    "illness":  [cd['hospital'], cd['patient'],     cd['admission']]}
                
                c_dict = {
                    "decision": [["hospital", "patient", "admission", "illness"], "decision", 'Total Costs', 'regression'],
                    "hospital": [["patient", "admission", "illness"], "hospital", 'Emergency Department Indicator_N', 'regression'],
                    "patient": [["hospital", "admission", "illness"],"patient", 'Age Group_70 or Older', 'regression'],
                    "admission": [["hospital", "patient", "illness"], "admission", 'Length of Stay', 'regression'],
                    "illness": [["hospital", "patient", "admission"],"illness", 'APR Risk of Mortality_Major', 'regression']
                }

            if app_name == "HIS17-onehot-random":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8 # maximum number of concepts to be created

                features_randomize = ['Length of Stay']

                d_name = "onehot_prep.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.7, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                # cd['c0'] = {'Facility Name_Staten Island University Hosp-North', 'Facility Name_Mount Sinai Beth Israel', 'Facility Name_Vassar Brothers Medical Center', 'Facility Name_University Hospital', 'Facility Name_The Unity Hospital of Rochester', 'Zip Code_100', 'Payment Typology 3_Self-Pay', 'Patient Disposition_Home w/ Home Health Services', 'Payment Typology 1_Private Health Insurance', 'Length of Stay', 'APR Medical Surgical Description_Medical'
                #     }

                # cd['c1'] = {'Facility Name_University Hospital - Stony Brook Southampton Hospital', 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'Facility Name_TLC Health Network Lake Shore Hospital', 'Facility Name_Wyckoff Heights Medical Center', 'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 'Facility Name_White Plains Hospital Center', 'Facility Name_Westchester Medical Center', 'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 'Gender_F', 'Gender_M', 'Patient Disposition_Home or Self Care', 'Payment Typology 2_Medicaid', 'Age Group_0 to 17', 'Ethnicity_Spanish/Hispanic', 'Payment Typology 1_Medicare', 'Payment Typology 2_Self-Pay', 'APR Severity of Illness Code_3', 'CCS Procedure Code_231', 'APR Medical Surgical Description_Not Applicable'
                #     }
                
                # cd['c2'] = {'Facility Name_University Hospital of Brooklyn', 'Emergency Department Indicator_N', 'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 'Facility Name_Syosset Hospital', 'Facility Name_Women And Childrens Hospital Of Buffalo', 'Facility Name_Westfield Memorial Hospital Inc', 'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 'Facility Name_Staten Island University Hosp-South', 'Facility Name_Strong Memorial Hospital', 'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 'Facility Name_The Burdett Care Center', 'Facility Name_Woodhull Medical & Mental Health Center', 'Facility Name_United Memorial Medical Center North Street Campus', 'Facility Name_UPSTATE University Hospital at Community General', 'Race_Other Race', 'Zip Code_104', 'Zip Code_112', 'Ethnicity_Not Span/Hispanic', 'Age Group_70 or Older', 'Age Group_50 to 69', 'Payment Typology 1_Medicaid', 'APR Severity of Illness Code_1', 'APR MDC Code_4', 'APR Risk of Mortality_Major', 'CCS Procedure Code_0'
                #     } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

                # cd['c3'] = {'Facility Name_University Hospital SUNY Health Science Center', 'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 'Facility Name_Wyoming County Community Hospital', 'Facility Name_UPMC Chautauqua at WCA', 'Facility Name_Mount Sinai Hospital', 'Hospital County_Bronx', 'Hospital County_Manhattan', 'Facility Name_St. Marys Healthcare', 'Emergency Department Indicator_Y', 'Age Group_30 to 49', 'Race_Black/African American', 'Age Group_18 to 29', 'Race_White', 'Type of Admission_Elective', 'Type of Admission_Emergency', 'APR Severity of Illness Code_2', 'APR Risk of Mortality_Minor', 'APR MDC Code_14', 'APR Medical Surgical Description_Surgical', 'APR Risk of Mortality_Moderate', 'APR MDC Code_5'
                #     }
                
                cd['c0']  = { 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'Facility Name_White Plains Hospital Center', 'Facility Name_Staten Island University Hosp-South', 'Facility Name_United Health Services Hospitals Inc. - Wilson Medical Center', 'Facility Name_University Hospital of Brooklyn', 'Facility Name_Montefiore Med Center - Jack D Weiler Hosp of A Einstein College Div', 'Emergency Department Indicator_Y', 'Payment Typology 1_Medicare', 'Race_Black/African American', 'Zip Code_112', 'Age Group_18 to 29', 'APR Risk of Mortality_Moderate', 'APR Medical Surgical Description_Medical' }
                cd['c1']  = { 'Hospital County_Bronx', 'Facility Name_UPSTATE University Hospital at Community General', 'Facility Name_University Hospital - Stony Brook Southampton Hospital', 'Facility Name_The University of Vermont Health Network - Champlain Valley Physicians Hospital', 'Facility Name_The Unity Hospital of Rochester', 'Facility Name_Montefiore Medical Center - Henry & Lucy Moses Div', 'Facility Name_Syosset Hospital', 'Hospital County_Manhattan', 'Facility Name_United Health Services Hospitals Inc. - Binghamton General Hospital', 'Facility Name_The University of Vermont Health Network - Elizabethtown Community Hospital', 'Facility Name_UPMC Chautauqua at WCA', 'Facility Name_Wyoming County Community Hospital', 'Ethnicity_Spanish/Hispanic', 'Race_White', 'Payment Typology 3_Self-Pay', 'Age Group_70 or Older', 'Ethnicity_Not Span/Hispanic', 'APR Risk of Mortality_Minor', 'APR Medical Surgical Description_Not Applicable', 'CCS Procedure Code_0' }
                cd['c2']  = { 'Facility Name_St. Marys Healthcare', 'Facility Name_University Hospital SUNY Health Science Center', 'Facility Name_Vassar Brothers Medical Center', 'Facility Name_Mount Sinai Beth Israel', 'Facility Name_United Memorial Medical Center North Street Campus', 'Facility Name_Wyckoff Heights Medical Center', 'Payment Typology 2_Medicaid', 'Gender_F', 'Type of Admission_Elective', 'APR Severity of Illness Code_1', 'APR MDC Code_14', 'APR MDC Code_5', 'APR Risk of Mortality_Major', 'CCS Procedure Code_231' }
                cd['c3']  = { 'Facility Name_Mount Sinai Hospital', 'Facility Name_Winifred Masterson Burke Rehabilitation Hospital', 'Emergency Department Indicator_N', 'Facility Name_Women And Childrens Hospital Of Buffalo', 'Facility Name_Westchester Medical Center', 'Facility Name_University Hospital', 'Facility Name_Staten Island University Hosp-North', 'Facility Name_Westfield Memorial Hospital Inc', 'Facility Name_Strong Memorial Hospital', 'Facility Name_TLC Health Network Lake Shore Hospital', 'Facility Name_Woodhull Medical & Mental Health Center', 'Facility Name_The Burdett Care Center', 'Payment Typology 1_Private Health Insurance', 'Patient Disposition_Home w/ Home Health Services', 'Race_Other Race', 'Zip Code_100', 'Patient Disposition_Home or Self Care', 'Zip Code_104', 'Payment Typology 1_Medicaid', 'Gender_M', 'Payment Typology 2_Self-Pay', 'Age Group_30 to 49', 'Age Group_0 to 17', 'Age Group_50 to 69', 'Length of Stay', 'Type of Admission_Emergency', 'APR Severity of Illness Code_3', 'APR Severity of Illness Code_2', 'APR Medical Surgical Description_Surgical', 'APR MDC Code_4' }

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'c0': cd['c0'], 
                    'c1': cd['c1'], 
                    'c2': cd['c2'], 
                    'c3': cd['c3']
                }

                predictors_all = []
                predictors_all.extend(cd['c0'])
                predictors_all.extend(cd['c1'])
                predictors_all.extend(cd['c2'])
                predictors_all.extend(cd['c3'])
                predictors_all.extend(cd['decision'])

                # f_dict = {
                # "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                # "hospital": [cd['patient'], cd['admission'], cd['illness']],
                # "patient": [cd['hospital'], cd['admission'], cd['illness']],
                # "admission": [cd['hospital'], cd['patient'], cd['illness']],
                # "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                f_dict = {
                    "decision": [cd['c0'], cd['c1'], cd['c2'], cd['c3']],
                    "c0":   [cd['c1'], cd['c2'], cd['c3']],
                    "c1":   [cd['c0'], cd['c2'], cd['c3']],
                    "c2":   [cd['c0'], cd['c1'], cd['c3']],
                    "c3":   [cd['c0'], cd['c1'], cd['c2']]}
                
                c_dict = {
                    "decision": [["c0", "c1", "c2", "c3"], "decision", 'Total Costs', 'regression'],
                    "c0": [["c1", "c2", "c3"], "c0", 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'regression'],
                    "c1": [["c0", "c2", "c3"], "c1", 'Hospital County_Bronx', 'regression'],
                    "c2": [["c0", "c1", "c3"], "c2", 'Facility Name_St. Marys Healthcare', 'regression'],
                    "c3": [["c0", "c1", "c2"], "c3", 'Facility Name_Mount Sinai Hospital', 'regression']
                }

            if app_name == "HIS17-two-classes":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8

                features_randomize = [] # ['Length of Stay']

                #d_name = "onehot_prep.csv"
                #d_name_short = "HIS17"
                #path_data = '~/Dropbox/CAData/'
                #d_name = "hdma_1M.csv"
                d_name = "hdma_1M_rankings.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}
                df['Total Costs'] = np.log1p(df['Total Costs'])


                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.5, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['c0']  = {
                    'Hospital Service Area', 
                    'Operating Certificate Number', 
                    'Permanent Facility Id',
                    'Facility Name',
                    'Gender',
                    'Ethnicity',
                    'Patient Disposition',
                    'Payment Typology 1',
                    'Payment Typology 2', 
                    'Payment Typology 3',
                    'Length of Stay', 
                    'Type of Admission',
                    'Emergency Department Indicator',
                    'CCS Diagnosis Description', 
                    'CCS Procedure Code'
                    }

                cd['c1']  = {
                    'APR DRG Code',
                    'APR MDC Description', 
                    'APR Risk of Mortality',
                    'APR Medical Surgical Description',
                    'Age Group', 
                    'Zip Code', 
                    'Hospital County',
                    'Race',
                    'APR Severity of Illness Code', 
                    'APR Severity of Illness Description', 
                    'APR DRG Description', 
                    'APR MDC Code',
                    'CCS Diagnosis Code', 
                    'CCS Procedure Description'
                    }

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'c0': cd['c0']
                }

                predictors_all = []
                predictors_all.extend(cd['c0'])
                predictors_all.extend(cd['c1'])
                predictors_all.extend(cd['decision'])

                # f_dict = {
                # "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                # "hospital": [cd['patient'], cd['admission'], cd['illness']],
                # "patient": [cd['hospital'], cd['admission'], cd['illness']],
                # "admission": [cd['hospital'], cd['patient'], cd['illness']],
                # "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                f_dict = {
                    "decision": [cd['c0'], cd['c1']],
                    "c0":   [cd['c1']],
                    "c1":   [cd['c0']]
                    }
                
                c_dict = {
                    "decision": [["c0", "c1"], "decision", 'Total Costs', 'regression'],
                    "c0": [["c1"], "c0", 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'regression'],
                    "c1": [["c0"], "c1", 'Hospital County_Bronx', 'regression'],
                }

            if app_name == "HIS17":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.05 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.1 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8

                #path_data = '../../data/'
                # ## Data Engineering
                #d_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2017.csv"
                #d_name_short = "HIS17"
                #df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')
                
                # path_data = '../../data/'

                #path_data = '~/Dropbox/CAData/'
                d_name = "hdma_1M.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.01, random_state=1)
                if scale == 1: data = df.sample(frac=0.2, random_state=1)
                if scale == 2: data = df.sample(frac=0.7, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['hospital'] = {'Hospital Service Area', 'Hospital County',
                    'Operating Certificate Number', 'Permanent Facility Id',
                    'Facility Name'}
                
                cd['patient'] = {'Age Group', 'Zip Code', 'Gender', 'Race',
                    'Ethnicity','Patient Disposition','Payment Typology 1',
                    'Payment Typology 2', 'Payment Typology 3'}
                
                cd['admission'] = {'Length of Stay', 'Type of Admission',
                    'Emergency Department Indicator'
                    } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

                cd['illness'] = {'CCS Diagnosis Code','CCS Diagnosis Description', 
                    'CCS Procedure Code', 'CCS Procedure Description', 
                    'APR DRG Code', 'APR DRG Description',
                    'APR MDC Code', 'APR MDC Description', 
                    'APR Risk of Mortality',
                    'APR Medical Surgical Description', 
                    'APR Severity of Illness Code', 'APR Severity of Illness Description'}

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'hospital': cd['hospital'], 
                    'patient': cd['patient'], 
                    'admission': cd['admission'], 
                    'illness': cd['illness']
                }

                predictors_all = []
                predictors_all.extend(cd['hospital'])
                predictors_all.extend(cd['patient'])
                predictors_all.extend(cd['admission'])
                predictors_all.extend(cd['illness'])
                predictors_all.extend(cd['decision'])

                f_dict = {
                "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                "hospital": [cd['patient'], cd['admission'], cd['illness']],
                "patient": [cd['hospital'], cd['admission'], cd['illness']],
                "admission": [cd['hospital'], cd['patient'], cd['illness']],
                "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                
                c_dict = {
                    "decision": [["hospital", "patient", "admission", "illness"], "decision", 'Total Costs', 'regression'],
                    "hospital": [["patient", "admission", "illness"], "hospital", 'Hospital Service Area', 'regression'],
                "patient": [["hospital", "admission", "illness"],"patient", 'Gender', 'regression'],
                "admission": [["hospital", "patient", "illness"], "admission", 'Length of Stay', 'regression'],
                "illness": [["hospital", "patient", "admission"],"illness", 'APR Risk of Mortality', 'regression']
            }
                
            if app_name == "HIS17-2":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.05 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.1 # 0.1 Threshold for R2 for predicting an attribute of a concept
                "HIS17-two-classes", "HIS17", 
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8

                features_randomize = ['Length of Stay']

                #path_data = '../../data/'
                # ## Data Engineering
                #d_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2017.csv"
                #d_name_short = "HIS17"
                #df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')
                
                # path_data = '../../data/'
                #path_data = '~/Dropbox/CAData/'
                d_name = "hdma_1M.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.0001, random_state=1)
                if scale == 1: data = df.sample(frac=0.2, random_state=1)
                if scale == 2: data = df.sample(frac=0.5, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['hospital'] = {'Hospital Service Area', 
                    'Operating Certificate Number', 'Permanent Facility Id',
                    'Facility Name'}
                
                cd['patient'] = {'Gender',
                    'Ethnicity','Patient Disposition','Payment Typology 1',
                    'Payment Typology 2', 'Payment Typology 3'}
                
                cd['admission'] = {'Length of Stay', 'Type of Admission',
                    'Emergency Department Indicator'
                    } #'total_charges', 'Abortion Edit Indicator','Discharge Year', 'Birth Weight'

                cd['illness'] = {'CCS Diagnosis Description', 
                    'CCS Procedure Code',
                    'APR DRG Code',
                    'APR MDC Description', 
                    'APR Risk of Mortality',
                    'APR Medical Surgical Description'
                    }
                #----
                cd['Demographic Information'] = {'Age Group', 
                                                    'Zip Code', 
                                                    'Hospital County',
                                                    'Race'}

                cd['Clinical Severity and Classification'] = {'APR Severity of Illness Code', 
                                                            'APR Severity of Illness Description', 
                                                            'APR DRG Description', 
                                                            'APR MDC Code'}

                cd['Procedural and Diagnosis Details'] = {'CCS Diagnosis Code', 
                                                        'CCS Procedure Description'}

                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'hospital': cd['hospital'], 
                    'patient': cd['patient'], 
                    'admission': cd['admission'], 
                    'illness': cd['illness'],
                    'Demographic Information': cd['Demographic Information'],
                    'Clinical Severity and Classification': cd['Clinical Severity and Classification'],
                    'Procedural and Diagnosis Details': cd['Procedural and Diagnosis Details']
                }

                predictors_all = []
                predictors_all.extend(cd['hospital'])
                predictors_all.extend(cd['patient'])
                predictors_all.extend(cd['admission'])
                predictors_all.extend(cd['illness'])
                predictors_all.extend(cd['Demographic Information'])
                predictors_all.extend(cd['Clinical Severity and Classification'])
                predictors_all.extend(cd['Procedural and Diagnosis Details'])
                predictors_all.extend(cd['decision'])

                f_dict = {
                "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness'], cd['Demographic Information'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "hospital": [cd['patient'], cd['admission'], cd['illness'], cd['Demographic Information'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "patient": [cd['hospital'], cd['admission'], cd['illness'], cd['Demographic Information'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "admission": [cd['hospital'], cd['patient'], cd['illness'], cd['Demographic Information'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "illness": [cd['hospital'], cd['patient'], cd['admission'], cd['Demographic Information'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "Demographic Information": [cd['hospital'], cd['patient'], cd['admission'], cd['illness'], cd['Clinical Severity and Classification'], cd['Procedural and Diagnosis Details']],
                "Clinical Severity and Classification": [cd['hospital'], cd['patient'], cd['admission'], cd['illness'], cd['Demographic Information'], cd['Procedural and Diagnosis Details']],
                "Procedural and Diagnosis Details": [cd['hospital'], cd['patient'], cd['admission'], cd['illness'], cd['Demographic Information'], cd['Clinical Severity and Classification']]
                }
                
                c_dict = {
                    "decision": [["hospital", "patient", "admission", "illness", "Demographic Information", "Clinical Severity and Classification", "Procedural and Diagnosis Details"], "decision", 'Total Costs', 'regression'],
                    "hospital": [["patient", "admission", "illness", "Demographic Information", "Clinical Severity and Classification", "Procedural and Diagnosis Details"], "hospital", 'Hospital Service Area', 'regression'],
                    "patient": [["hospital", "admission", "illness", "Demographic Information", "Clinical Severity and Classification", "Procedural and Diagnosis Details"],"patient", 'Gender', 'regression'],
                    "admission": [["hospital", "patient", "illness", "Demographic Information", "Clinical Severity and Classification", "Procedural and Diagnosis Details"], "admission", 'Length of Stay', 'regression'],
                    "illness": [["hospital", "patient", "admission", "Demographic Information", "Clinical Severity and Classification", "Procedural and Diagnosis Details"],"illness", 'APR Risk of Mortality', 'regression'],
                    "Demographic Information": [["hospital", "patient", "admission", "illness", "Clinical Severity and Classification", "Procedural and Diagnosis Details"], "Demographic Information", 'Age Group', 'regression'],
                    "Clinical Severity and Classification": [["hospital", "patient", "admission", "illness", "Demographic Information", "Procedural and Diagnosis Details"], "Clinical Severity and Classification", 'APR Severity of Illness Code', 'regression'],
                    "Procedural and Diagnosis Details": [["hospital", "patient", "admission", "illness", "Demographic Information", "Clinical Severity and Classification"], "Procedural and Diagnosis Details", 'CCS Diagnosis Code', 'regression']
                }
      
            if app_name == "LOAN":
                #scale = 1 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.05 # 0.1 negative and positive threshold for relations
                r2_threshold = 0.02 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 2 # for HIS: 4 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.1 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 10

                features_randomize = []

                # ## Data Engineering
                #d_name = "xaa.csv"
                d_name = "loan_10_Percent.csv"
                d_name_short = "LOAN"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'action_taken'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1) # frac=0.001 good results
                if scale == 1: data = df.sample(frac=0.2, random_state=1)
                if scale == 2: data = df.sample(frac=0.5, random_state=1)
                if scale == 3: data = df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                # cd['applicant'] = {'applicant_sex', 'purchaser_type', 'applicant_age'} # , 'income', 'county_code' 
                # cd['loanapplication'] = {'loan_term', 'origination_charges', 'derived_loan_product_type'} # 'property_value'
                # cd['loan'] = {'loan_amount',  'total_loan_costs'} # 'rate_spread','total_charges', 'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose'
                # cd['decision'] = {'action_taken'}

                cd['applicant'] = {'applicant_sex', 'purchaser_type'} # 'applicant_age', 'derived_loan_product_type','lien_status', 'preapproval',
                cd['loanapplication'] = {'loan_term', 'origination_charges',  'property_value'}
                cd['loan'] = {'loan_amount',  'total_loan_costs', 'rate_spread', 'lien_status', 'hoepa_status', 'interest_rate',  'loan_type', 'loan_purpose', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose'}
                cd['decision'] = {'action_taken'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'applicant': cd['applicant'], 
                    'loanapplication': cd['loanapplication'], 
                    'loan': cd['loan']
                }

                predictors_all = []
                predictors_all.extend(cd['applicant'])
                predictors_all.extend(cd['loanapplication'])
                predictors_all.extend(cd['loan'])
                predictors_all.extend(cd['decision'])

                f_dict = {
                    "decision": [cd['applicant'], cd['loanapplication'], cd['loan']],
                    "applicant": [cd['loanapplication'], cd['loan']],
                    "loanapplication": [cd['applicant'], cd['loan']],
                    "loan": [cd['applicant'], cd['loanapplication']]
                    }
                
                c_dict = {
                    "decision": [["applicant", "loanapplication", "loan"], "decision", 'action_taken', 'regression'],
                    "applicant": [["loanapplication", "loan"], "applicant", 'applicant_age', 'regression'],
                    "loanapplication": [["applicant", "loan"],"loanapplication", 'origination_charges', 'regression'],
                     "loan": [["applicant", "loanapplication"], "loan", 'loan_amount', 'regression']
                 }

            if app_name == "LOAN-2":
                #scale = 1 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.05 # 0.1 negative and positive threshold for relations
                r2_threshold = 0.02 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 2 # for HIS: 4 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.1 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 10

                features_randomize = []

                # ## Data Engineering
                d_name = "xaa.csv"
                d_name_short = "LOAN"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'action_taken'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1) # frac=0.001 good results
                if scale == 1: data = df.sample(frac=0.2,   random_state=1)
                if scale == 2: data = df.sample(frac=0.5,   random_state=1)
                if scale == 3: data = df.sample(frac=1.0,   random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                # cd['applicant'] = {'applicant_sex', 'purchaser_type', 'applicant_age'} # , 'income', 'county_code' 
                # cd['loanapplication'] = {'loan_term', 'origination_charges', 'derived_loan_product_type'} # 'property_value'
                # cd['loan'] = {'loan_amount',  'total_loan_costs'} # 'rate_spread','total_charges', 'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose', 'lien_status', 'hoepa_status', 'interest_rate', 'preapproval', 'loan_type', 'loan_purpose'
                # cd['decision'] = {'action_taken'}

                cd['c_0']  = { 'applicant_age', 'derived_loan_product_type', 'origination_charges', 'rate_spread', 'loan_amount', 'loan_purpose', 'loan_type', 'lien_status', 'interest_rate' }
                cd['c_1']  = { 'purchaser_type', 'applicant_sex', 'loan_term', 'property_value', 'hoepa_status', 'total_loan_costs', 'preapproval' }
                cd['decision'] = {'action_taken'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'c_0': cd['c_0'], 
                    'c_1': cd['c_1'] 
                }

                predictors_all = []
                predictors_all.extend(cd['c_0'])
                predictors_all.extend(cd['c_1'])
                predictors_all.extend(cd['decision'])

                f_dict = {
                    "decision": [cd['c_0'], cd['c_1']],
                    "c_0": [cd['c_1']],
                    "c_1": [cd['c_0']]
                    }
                
                c_dict = {
                    "decision": [["c_0", "c_1"], "decision", 'action_taken', 'regression'],
                    "c_0": [["c_1"], "c_0", 'applicant_age', 'regression'],
                    "c_1": [["c_0"],"c_1", 'purchaser_type', 'regression'],
                 }

            if app_name == "HIS17-random":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8

                features_randomize = ['Length of Stay']

                #d_name = "onehot_prep.csv"
                #d_name_short = "HIS17"
                #path_data = '~/Dropbox/CAData/'
                #d_name = "hdma_1M.csv"
                d_name = "hdma_1M_rankings.csv"
                d_name_short = "HIS17"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Total Costs'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.5, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                # cd['c0']  = {"Race", "Payment Typology 3", "APR DRG Description", "APR Medical Surgical Description", "APR Severity of Illness Description"}
                # cd['c1']  = {"Payment Typology 2", "APR MDC Code", "Permanent Facility Id", "Payment Typology 1"}
                # cd['c2']  = {"APR MDC Description", "Operating Certificate Number", "Facility Name", "Zip Code", "Type of Admission"}
                # cd['c3']  = {"Hospital County", "CCS Diagnosis Code", "CCS Procedure Description", "Ethnicity", "Hospital Service Area"}
                # cd['c4']  = {"APR Severity of Illness Code", "APR Risk of Mortality", "Emergency Department Indicator"}
                # cd['c5']  = {"APR DRG Code", "CCS Diagnosis Description", "Length of Stay", "Age Group"}


                #cd['c0']  = {"Zip Code", "CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Age Group", "Permanent Facility Id", "Payment Typology 2", "APR DRG Description", "APR DRG Code", "Payment Typology 3"}
                #cd['c1']  = {"Payment Typology 1", "APR MDC Description", "Hospital County", "Race", "Length of Stay", "Hospital Service Area", "Facility Name"}
                #cd['c2']  = {"Type of Admission", "Emergency Department Indicator", "APR Severity of Illness Description", "Ethnicity", "APR Severity of Illness Code", "APR Risk of Mortality", "Operating Certificate Number", "APR Medical Surgical Description", "CCS Diagnosis Description"}

                cd['c0']  = {"Zip Code", "Permanent Facility Id", "Hospital County", "Race", "Ethnicity", "Age Group", "Length of Stay", "Hospital Service Area", 
                             "Facility Name", "Payment Typology 1", "APR MDC Description", "Emergency Department Indicator", "APR Severity of Illness Description", 
                             "APR Risk of Mortality", "CCS Diagnosis Description", "Gender", "Patient Disposition", "CCS Procedure Code"}
                cd['c1']  = {"CCS Procedure Description", "CCS Diagnosis Code", "APR MDC Code", "Payment Typology 2", "APR DRG Description", 
                             "APR DRG Code", "Payment Typology 3", "Type of Admission", "APR Severity of Illness Code", "Operating Certificate Number", 
                             "APR Medical Surgical Description"}
                
                cd['decision'] = {'Total Costs'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'c0': cd['c0'],
                    'c1': cd['c1']
                }

                predictors_all = []
                predictors_all.extend(cd['c0'])
                predictors_all.extend(cd['c1'])
                predictors_all.extend(cd['decision'])

                f_dict = {
                    "decision": [cd['c0'], cd['c1']],
                    "c0":   [cd['c1']],
                    "c1":   [cd['c0']]
                    }
                
                c_dict = {
                    "decision": [["c0", "c1"], "decision", 'Total Costs', 'regression'],
                    "c0": [["c1"], "c0", 'Zip Code', 'regression'],
                    "c1": [["c0"], "c1", 'CCS Procedure Description', 'regression']
                }

            if app_name == "HIS17-two-classes-d":

                #scale = 3 # size of the dataset fraction
                step = 1 # configuration
                #epsilon = 0.01 # 0.03 negative and positive threshold for relations
                r2_threshold = 0.01 # 0.1 Threshold for R2 for predicting an attribute of a concept
                
                cut = 3 # cut for sep = 1, i.e., number of negative features for an input concept
                cluster_cut = 0.01 # 0.1 cut for TSKmeans inertia_ value; everything smaller means that clusters are too small
                ci = 1 # create concept index
                #concept_max = 8

                features_randomize = ['Length of Stay']

                #d_name = "onehot_prep.csv"
                #d_name_short = "HIS17"
                #path_data = '~/Dropbox/CAData/'
                #d_name = "hdma_1M.csv"
                d_name = "hdma_1M_d_rankings.csv"
                d_name_short = "HIS17-d"
                df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding = 'latin1')

                non_input_features = {'Gesamtkosten'}

                if scale == 0: data = df.sample(frac=0.001, random_state=1)
                if scale == 1: data = df.sample(frac=0.1, random_state=1)
                if scale == 2: data = df.sample(frac=0.5, random_state=1)
                if scale == 3: data = df # df.sample(frac=1.0, random_state=1)

                #CONFIGURATION
                
                non_input_concepts = ['decision']

                cd = {}

                cd['c0']  = {
                    'Aufenthaltsdauer','Versorgungsgebiet','Landkreis','Krankenhausname','Altersgruppe','PLZ','Geschlecht','Staatsangehoerigkeit',
                    'Ethnie','Art der Aufnahme','Patientenstatus bei Entlassung','Diagnosebeschreibung nach CCS ICD','Prozedurbeschreibung nach CCS OPS',
                    'DRGBeschreibung','MDCAehnliche Gruppen im DRG','Medizinische Hauptdiagnose Beschreibung','Schweregrad Code'
                    }

                cd['c1']  = {
                    'Schweregrad der Erkrankung Beschreibung','Sterblichkeitsrisiko','MedizinischChirurgische Beschreibung',
                    'Zahlungsart 1','Zahlungsart 2','Zahlungsart 3','Indikator fuer Abtreibungsbearbeitung','Notaufnahme Indikator','Betriebsstaettennummer',
                    'Krankenhausnummer','ICDDiagnoseschluessel','OPSCode','DRGCode'
                    }

                cd['decision'] = {'Gesamtkosten'}

                concepts_all = {
                    'decision': cd['decision'], 
                    'c0': cd['c0']
                }

                predictors_all = []
                predictors_all.extend(cd['c0'])
                predictors_all.extend(cd['c1'])
                predictors_all.extend(cd['decision'])

                # f_dict = {
                # "decision": [cd['hospital'], cd['patient'], cd['admission'], cd['illness']],
                # "hospital": [cd['patient'], cd['admission'], cd['illness']],
                # "patient": [cd['hospital'], cd['admission'], cd['illness']],
                # "admission": [cd['hospital'], cd['patient'], cd['illness']],
                # "illness": [cd['hospital'], cd['patient'], cd['admission']]}
                f_dict = {
                    "decision": [cd['c0'], cd['c1']],
                    "c0":   [cd['c1']],
                    "c1":   [cd['c0']]
                    }
                
                c_dict = {
                    "decision": [["c0", "c1"], "decision", 'Gesamtkosten', 'regression'],
                    "c0": [["c1"], "c0", 'Facility Name_St. Marys Healthcare - Amsterdam Memorial Campus', 'regression'],
                    "c1": [["c0"], "c1", 'Hospital County_Bronx', 'regression'],
                }


            # copy cd to cd_init
            cd_init = cd.copy()

            # if folder does not exist create it
            if not os.path.exists(path_results):
                os.makedirs(path_results)
                print("Created folder: ", path_results)

            # save cd_init to csv file
            cd_df = pd.DataFrame.from_dict(cd_init, orient='index')
            cd_df.to_csv(path_results + 'cd_0.csv')

            cleanup_files(path_results)

            create_directories([path_results, path_figures, path_graphs])

            # make list ai_list with 'iter' as first element
            ai_list = ['main_iter', 'iter', 'concept']

            ai_list.extend(predictors_all)

            # initialize attribute_iterations with ai_list and initialize index of attribute_iterations with ascending integers
            attribute_iterations = pd.DataFrame(columns=ai_list)

            # if attribute_iterations.csv does not exist create and save attribute_iterations to csv file with column names
            if not os.path.exists(path_results + 'attribute_iterations.csv'):
                attribute_iterations.to_csv(path_results + 'attribute_iterations.csv', header=True, index=False)
            
            idx_ai = 0
            c_max = concept_max

            concept_counter = Counter2()
                
            #c1 = 1

            GCBM = pd.DataFrame({
                'from':[],
                'to' :[],
                'gcbm':[],
                'iteration': []
            })
            
            i_id = 0
            cluster_id = 1
            eps = 0.3 # epsilon value for change in the separation index between two iterations
            si = 10 # arbitrarily chosen but needs to obey eps
            si_new = 9 # dito
            
            sep_feats = [x for x in range(len(data.columns))]
            #concepts_all = {}

            print(data.columns)

            # 1. Data Pre-Processing
            feat_data = pre_processing(app_name, data, predictors_all, features_randomize)

            # save feat_data to csv file
            feat_data.to_csv(path_results + 'feat_data-' + app_name + '.csv')

            print(feat_data.columns)

            results = model_train_evaluate(feat_data, c_dict['decision'][2], model_path)

            print(results)

            # Randomize the outcome feature in c_dict
            c_dict = randomize_outcome(c_dict, cd)

            cd_df = pd.DataFrame.from_dict(cd, orient='index')

            # add entry to file 'concept_definitions_all-' + str(mi) + '.csv' with mi
            # with open(path_results + 'concept_definitions_all-' + str(mi) + '.csv', 'a') as file:
            #     file.write(f"After pre-processing: {mi} {i_id}\n")

            # append cd_df to csv file
            cd_df.to_csv(path_results + 'concept_definitions_all-' + str(mi) + '.csv', header=False, mode = 'a')

            c_number = 1
            print(f"calignment: {c_number}")
            # 1st calignment
            # delete features with low shap values
            r2_all, GCBM, shap_values, iso, del_features, cd, concepts_all, predictors_all, feat_data = calignment(
                c_dict, 
                f_dict, 
                feat_data, 
                cd, 
                concepts_all,
                predictors_all,
                mi,
                i_id, 
                GCBM, 
                d_name_short, 
                path_results,
                path_figures,
                path_graphs,
                non_input_concepts, 
                epsilon,
                eps_threshold,
                r2_threshold,
                c_number,
                prediction_mode=prediction_mode
                )

            i_id += 1

            # cd, predictors_all, feat_data = delete_low_shap(cd, predictors_all, feat_data, del_features, path_results, mi)
            cd_df = pd.DataFrame.from_dict(cd, orient='index')
            cd_df.to_csv(path_results + 'concept_definitions-' + str(mi) + '.csv')

            # append cd_df to csv file
            # add entry to file 'concept_definitions_all-' + str(mi) + '.csv' with mi
            # with open(path_results + 'concept_definitions_all-' + str(mi) + '.csv', 'a') as file:
            #     file.write(f"after first CA: {mi}\n")
            # cd_df.to_csv(path_results + 'concept_definitions_all-' + str(mi) + '.csv', header=False, mode = 'a')

            # iterate cd and sum up items
            num_items = 0
            for key, value in cd.items():
                num_items += len(value)

            # save del_features to csv file
            del_df = pd.DataFrame(del_features)
            del_df.to_csv(path_results + 'del_features.csv', index=False)

            # save predictors_all to csv file
            pa_df = pd.DataFrame(predictors_all)
            pa_df.to_csv(path_results + 'predictors_all.csv', index=False)

            # Randomize the outcome feature in c_dict
            c_dict = randomize_outcome(c_dict, cd)
            
            print("\n ================== Main loop: ", mi, "Iteration: ", i_id, "======================\n")

            c_number = 2
            print(f"calignment: {c_number}")
            # 2nd calignment
            r2_all, GCBM, shap_values, iso, del_features, cd, concepts_all, predictors_all, feat_data = calignment(c_dict, 
                                f_dict, 
                                feat_data, 
                                cd, 
                                concepts_all,
                                predictors_all, 
                                mi,
                                i_id, 
                                GCBM, 
                                d_name_short, 
                                path_results,
                                path_figures,
                                path_graphs,
                                non_input_concepts, 
                                epsilon,
                                eps_threshold, 
                                r2_threshold,
                                c_number,
                                prediction_mode=prediction_mode)
            
            print(cd)

            #cd, predictors_all, feat_data = delete_low_shap(cd, predictors_all, feat_data, del_features, path_results, mi)
            cd_df = pd.DataFrame.from_dict(cd, orient='index')
            cd_df.to_csv(path_results + 'concept_definitions-' + str(mi) + '.csv')

            # append cd_df to csv file
            # add entry to file 'concept_definitions_all-' + str(mi) + '.csv' with mi
            # with open(path_results + 'concept_definitions_all-' + str(mi) + '.csv', 'a') as file:
            #     file.write(f"After 2nd calignment: {mi}\n")
            # cd_df.to_csv(path_results + 'concept_definitions_all-' + str(mi) + '.csv', header=False, mode = 'a')

            # iterate cd and sum up items
            num_items = 0
            for key, value in cd.items():
                num_items += len(value)

            r2 = pd.DataFrame.from_dict(r2_all, orient='index')
            #r2 = r2.transpose()
            r2.to_csv(path_results + 'r2.csv', mode = 'a')
            
            #sep_feats, ts_X = seperate_feature(cd, non_input_concepts, path_results, path_figures, epsilon, cluster_cut)
            
            if c_max > 0:
                # Instantiate FeatureSeparator with the required arguments
                separator = FeatureSeparator(cd, non_input_concepts,path_results, path_figures, epsilon, cluster_cut, c_max)

                # Call separate_features to perform the separation
                sep_feats, ts_X = separator.separate_features(attribute_iterations, mi, i_id, idx_ai)
                print(sep_feats)

                sf = pd.DataFrame(sep_feats)
                sf.to_csv(path_results +'sep_feats.csv', index=False)
                sf.to_csv(path_results +'sep_feats_hist.csv', index=False, mode='a')

                diff_sep_features = len(sep_feats)

                print("\nDifference in sep features before loop: ", diff_sep_features)
                
                c_all = concepts_all.copy()

                # set sep_feats to all values in cd except the last one
                # sep_feats = [x for x in range(len(data.columns))]

                print(sep_feats)

                sf = pd.DataFrame(sep_feats)
                sf.to_csv(path_results + 'sep_feats.csv', index=False)
                sf.to_csv(path_results + 'sep_feats_hist.csv', index=False, mode='a')

                diff_sep_features = len(sep_feats)

                print("\nDifference in sep features before loop: ", diff_sep_features)
                
                c_all = concepts_all.copy()
            
            # iteration without test for isomorphism
            while ((not iso or prediction_mode) and diff_sep_features > 0 and c_max > 0):

            # iteration with test for isomorphism
            #while (not iso and diff_sep_features > 0 and concept_max > 0):

                i_id += 1
                idx_ai += 1

                print("\n================== Main loop: ", mi ," Iteration: ", i_id, "======================\n")

                cd, ci, c_all, c_max = create_concepts(c_all, predictors_all, f_dict, c_dict, 
                                    non_input_concepts, non_input_features, sep_feats, 
                                    ts_X, cd, ci, cluster_cut, path_results,path_figures, concept_counter, c_max)
                    

                print("c_max: ", c_max)

                cd_df = pd.DataFrame.from_dict(cd, orient='index')

                # copy cd to cd_init
                cd_init = cd.copy()

                # save cd_init to csv file
                cd_df = pd.DataFrame.from_dict(cd_init, orient='index')
                cd_df.to_csv(path_results + 'cd_' + str(i_id) + '.csv')    

                with open(path_results + 'concept_definitions-' + str(mi) + '.csv', 'a') as file:
                    file.write("*iteration*\n")

                cd_df.to_csv(path_results + 'concept_definitions-' + str(mi) + '.csv', header=False, mode = 'a')

                # add entry to file 'concept_definitions_all-' + str(mi) + '.csv' with mi
                with open(path_results + 'concept_definitions_all-' + str(mi) + '.csv', 'a') as file:
                    file.write(f"With added concepts in while loop: {mi} {i_id}\n")

                # append cd_df to csv file
                cd_df.to_csv(path_results + 'concept_definitions_all-' + str(mi) + '.csv', header=False, mode = 'a')
                
                c_number = 3
                print(f"calignment: {c_number}")
                # 3rd calignment in while loop
                r2_all, GCBM, shap_values, iso, del_features, cd, concepts_all, predictors_all, feat_data = calignment(
                    c_dict, 
                    f_dict, 
                    feat_data, 
                    cd, 
                    concepts_all, 
                    predictors_all, 
                    mi, 
                    i_id, 
                    GCBM, 
                    d_name_short, 
                    path_results, 
                    path_figures, 
                    path_graphs, 
                    non_input_concepts, 
                    epsilon, 
                    eps_threshold,
                    r2_threshold, 
                    c_number, 
                    prediction_mode=prediction_mode
                    )

                #cd, predictors_all, feat_data = delete_low_shap(cd, predictors_all, feat_data, del_features, path_results, mi)
                cd_df = pd.DataFrame.from_dict(cd, orient='index')
                cd_df.to_csv(path_results + 'concept_definitions-' + str(mi) + '.csv')

                # add entry to file 'concept_definitions_all-' + str(mi) + '.csv' with mi
                # with open(path_results + 'concept_definitions_all-' + str(mi) + '.csv', 'a') as file:
                #     file.write(f"After 3rd calignment: {mi} {i_id}\n")

                # append cd_df to csv file
                # cd_df.to_csv(path_results + 'concept_definitions_all-' + str(mi) + '.csv', header=False, mode = 'a')

                r2 = pd.DataFrame.from_dict(r2_all, orient='index')
                #r2 = r2.transpose()
                r2.to_csv(path_results + 'r2.csv', mode = 'a')

                #read csv if exists; if not create it and read it into r2_df and initialize with r2.columns
                if os.path.exists(path_results + 'r2_t_hist.csv'):
                    r2_df = pd.read_csv(path_results + 'r2_t_hist.csv', delimiter=',',
                                        quotechar='"', encoding='latin1')
                else:
                    r2_df = pd.DataFrame(columns=r2.columns)
                    r2_df.to_csv(path_results + 'r2_t_hist.csv')

                r2_t = r2.transpose()

                # append dataframe r2_df to dataframe r2_t and write to csv
                r2_t = pd.concat([r2_t,r2_df], ignore_index=True)
                #r2_t = r2_t.append(r2_df, ignore_index=True)
                r2_t.to_csv(path_results + 'r2_t_hist.csv')

                sep_feats_mem = sep_feats.copy()

                sf = pd.DataFrame(sep_feats)
                sf.to_csv(path_results + 'sep_feats_mem.csv', index=False)
                
                # Instantiate FeatureSeparator with the required arguments
                separator = FeatureSeparator(cd, non_input_concepts, path_results, path_figures, epsilon, cluster_cut)

                # Call separate_features to perform the separation
                sep_feats, ts_X = separator.separate_features(attribute_iterations, mi, i_id, idx_ai)

                #merge sep_feats_mem and sep_feats
                sep_feats = sep_feats_mem + sep_feats

                sf = pd.DataFrame(sep_feats)
                sf.to_csv(path_results + 'sep_feats.csv', index=False)
                sf.to_csv(path_results + 'sep_feats_hist.csv', index=False, mode='a')

                GCBM.to_csv(path_results + 'GCBM.csv')

                diff_sep_features = len(sep_feats) - len(sep_feats_mem)
                print("\nDifference in sep features end loop: ", diff_sep_features)

            # 3. Post-Processing

            if os.path.exists(path_results + 'r2_t_hist.csv'):
                # write r2 to r2_t_hist.csv
                r2.to_csv(path_results + 'r2_t_hist.csv', mode='a')

            else:
                #("r2_t_hist.csv does not exist")
                #r2_df = pd.DataFrame(columns=r2.columns)
                r2.to_csv(path_results + 'r2_t_hist.csv')

            #ca_goodness_of_fit(r2_df)

            #r2_plot(r2_df, cluster_id, path_results, path_figures)

            fname = "path_results + 'concept_definitions_"+str(cluster_cut)+".csv"

            # delete '/' at the end of path_results and store in base_path
            base_path = path_results[:-1]

            if i_id > 1:
                execute_analysis(base_path, 'GCBM.csv', 'r2_t_hist.csv', 'concept_definitions-' + str(mi) + '.csv', "GCBM.txt")


            ci = 1
            concept_counter.reset()


        print("\n**** C'est fini! [Cluster cut: ", cluster_cut, "] ****")

        #logger.info('Finished')
            
######################################################    

# if __name__ == "__main__":

#     main()

#     print("end of main")

######################################################    

if __name__ == "__main__":
    
    main()

    print("end of main")