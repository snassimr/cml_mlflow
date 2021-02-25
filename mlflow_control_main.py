####################################################################################### Classification
####################################################################################### Classification - Setup
import os
import sys
import numpy  as np
import pandas as pd
import re
import logging

display_settings = {'display.max_rows' : 50 , 'display.max_columns' : 50 , 'display.width' : 200}
for op,value in display_settings.items():
    pd.set_option(op,value)

SYS_CML_DIR       = "C:\\Users\\nmatatov\\OneDrive - NI\\cml_workdir"

if (not(SYS_CML_DIR in sys.path)):
   sys.path.insert(0,SYS_CML_DIR)

from cml_mlt_setup_classification import *   
   
####################################################################################### Classification - Prepare Sample Data
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
bc_data        = load_breast_cancer()
df_bc = pd.DataFrame(bc_data.data,columns=bc_data.feature_names)
bc_data_target = ['M' if i == 0 else 'B' for i in list(bc_data.target)]
df_bc['Diagnosis'] = bc_data_target
df_bc.insert(0,'ID', ['I' + str(i)  for i in np.arange(len(df_bc))+1])
df_bc.set_index("ID" , inplace=True)
df_bc.insert(0,'TS',pd.date_range(end='1/11/2019', periods=len(df_bc), freq='D'))
# df_bc.insert(1,'GROUP', np.repeat(['A','B'], [round(df_bc.shape[0]*0.3) , round(df_bc.shape[0]*0.7)], axis=0))

modeling_input_data = df_bc

# Create less predictive data
from cml_mlt_setup_classification import *
# modeling_input_data.drop(['worst texture' , 'worst radius', 'worst concave points' , 'worst radius'] , axis = 1 , inplace = True)
modeling_input_data = modeling_input_data[APP_PARAMS['APP_METADATA_LIST'] + ['mean radius', 'mean texture', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry']]

################################################################################### Classification - Model Development
from cml_mlt_setup_classification import *
from cml_control_data_preparation import *
import cml.cml_modeling.cml_modeling_utils as cml_modeling_utils
       
SYS_MODELING_DIR            = APP_PARAMS['SYS_MODELING_DIR']
APP_PROJECT_ID              = APP_PARAMS['APP_PROJECT_ID']
APP_ML_TASK                 = APP_PARAMS['APP_ML_TASK']
APP_METADATA_LIST           = APP_PARAMS['APP_METADATA_LIST']
APP_D_IDENT                 = APP_PARAMS['APP_D_IDENT']
B_D_IDENT_INDEX             = APP_PARAMS['B_D_IDENT_INDEX']
APP_B_IDENT                 = APP_PARAMS['APP_B_IDENT']
APP_TS_FEATURE              = APP_PARAMS['APP_TS_FEATURE']
APP_TARGET_FEATURE          = APP_PARAMS['APP_TARGET_FEATURE']
APP_POSITIVE_TARGET_VALUE   = APP_PARAMS['APP_POSITIVE_TARGET_VALUE']

logger = set_logger(APP_PARAMS)

logger.info("Start experimentation design")

exp_design     = cml_modeling_utils.create_classification_exp_design(modeling_input_data , APP_D_IDENT , B_D_IDENT_INDEX , APP_TARGET_FEATURE , APP_EXP_CONTROL , APP_EXP_PARAMS)
exp_design_map = exp_design['exp_map']

ML_RUN_ID = "_".join([APP_PROJECT_ID , APP_TARGET_FEATURE , APP_MLA_ID])

exp_run_map = exp_design_map.copy()
exp_run_map.rename(columns = { 'training' : 'TRAINING_DATA_ID' , 'evaluation' : 'EVALUATION_DATA_ID'} , inplace = True)
exp_run_map.insert(0, 'PRED_MODEL_ID' , exp_run_map['TRAINING_DATA_ID'].apply(lambda x : ML_RUN_ID + "_" + x)) 

logger.info("End experimentation design")


