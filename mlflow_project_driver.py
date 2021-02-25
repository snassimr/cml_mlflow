# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:10:04 2021

@author: NMatatov
"""

import mlflow

mlflow.projects.run(
    'https://github.com/snassimr/cml_mlflow', 
    backend = 'local', 
    synchronous = False
    )

import mlflow
MLFLOW_URI = "C:\\Users\\nmatatov\\OneDrive - NI\\cml_mlflow"
mlflow.projects.run(MLFLOW_URI, experiment_name = None, experiment_id = None, run_id = None , entry_point = None, version = None, 
                    parameters  = None, 
                    docker_args = None, backend = 'local', backend_config = None, use_conda = True, storage_dir = None, synchronous=True)



import mlflow

mlflow.projects.run(
    'https://github.com/snassimr/celeb-cnn-project', 
    backend='local', 
    synchronous=False,
    parameters={
        'batch_size': 32, 
        'epochs': 10, 
        'convolutions': 3,
        'training_samples': 15000,
        'validation_samples': 2000,
        'randomize_images': True
    })