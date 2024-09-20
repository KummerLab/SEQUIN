# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:35:24 2024

@author: andrew.sauerbeck
"""
#########################################
## Load required packages
#########################################

import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import dask.dataframe
import numpy as np
import pandas as pd
import os
import scipy as scipy 
from matplotlib import gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import re

import multiprocessing as mp
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import wrap_non_picklable_objects
import time
from joblib import dump, load
import lightgbm as lgb
#%%
#########################################
## User defined required input parameters
#########################################
####################################################
## Define input and output folders
####################################################
## Output folder
out_folder = 'C:/Output_folder/'
## Input folder. This is the output from running the initial SEQUIN_v15 code.
input_folder  = 'C:/Input_Folder/'
## Output folder for LightGBM classified synaptic loci.
output_folder = input_folder + 'compiled/'

####################################################
## Output file name
####################################################
## Custom designed workflow based on Tilescans
## For code to work, out_file simply must be created
## Slide number
slide = 1
## Sectoion number
section = 3
## Compiled output file name
out_file = 'slide_' + str(slide) + '_sec_' + str(section) + '_lgbm_data.csv'





####################################################
## Pertrained LightGBM model to load
####################################################
## Model location and file name
pretrained_LightGBM_model = 'C:/SEQUINv15_LightGBM_model_1.joblib'



#%%
###############################################################
## Core code
## Just run this block of code.
## Do not edit.
##############################################################


input_files = np.sort(os.listdir(input_folder))
metadata_files = np.asarray([x for x in input_files if x.endswith('PSD_SYN_metadata.csv')] )
radii_files = np.asarray([x for x in input_files if x.endswith('3rd_channel_vRadii.csv')] )
neurite_files = np.asarray([x for x in input_files if x.endswith('3rd_channel_vRadii.csv')] )
subfield_files = np.asarray([x for x in input_files if x.endswith('subfield_neurite_data.csv')] )



model_reload = load(pretrained_LightGBM_model) 
total_data = [None] * np.shape(metadata_files)[0]

def get_data(i):
    file = metadata_files[i]
    file_start =   re.split('_PSD_SYN_metadata.csv',file)[0]
    current_metadata = np.asarray(dask.dataframe.read_csv(input_folder + file, delimiter=',', header=None)) 
    current_radii = np.asarray(dask.dataframe.read_csv(input_folder + file_start + '_3rd_channel_vRadii.csv', delimiter=',', header=None)) 
    current_neurite_positive = np.asarray(dask.dataframe.read_csv(input_folder + file_start + '_neurite_positive_data.csv', delimiter=',', header=None)) 
    current_subfield = np.asarray(dask.dataframe.read_csv(input_folder + file_start + '_subfield_neurite_data.csv', delimiter=',', header=None)) 
    
      
    current_metadata = current_metadata[1::,1::]
    current_radii = current_radii[1::,1::]
    current_neurite_positive = current_neurite_positive[1::,1::]
    current_subfield = current_subfield[1::,1::]
       
    
    tile_ID = (re.search('MT_' + '(.*?)' + '_PSD', file))
    tile_ID = tile_ID.group(1)
    tile_ID = np.repeat(tile_ID,np.shape(current_metadata)[0]).reshape(np.shape(current_metadata)[0],1)
 
    
    
    non_neurite_intensity = np.repeat(current_subfield[1,10],np.shape(current_metadata)[0]).reshape(np.shape(current_metadata)[0],1)
    
    current_neurite_positive_variables = current_neurite_positive[0,:]
    current_neurite_positive = current_neurite_positive[1::,:]
    
    current_radii_variables = current_radii[0:5,:]
    current_radii = current_radii[5::,:]
    
    psd_id_test_neurite = current_metadata[:,2].astype(float) - current_neurite_positive[:,0].astype(float)
    psd_id_test_radii = current_metadata[:,2].astype(float) - current_radii[:,0].astype(float)
    
    syn_id_test_neurite = current_metadata[:,435].astype(float) - current_neurite_positive[:,8].astype(float)
    syn_id_test_radii = current_metadata[:,435].astype(float) - current_radii[:,1].astype(float)
    
    psd_syn_id_test = np.unique([psd_id_test_neurite,psd_id_test_radii,syn_id_test_neurite,syn_id_test_radii])
    if psd_syn_id_test == 0:
        print('spot ids match!')
    
    
    if psd_syn_id_test != 0:
        print('spot ids dont match!')
        return
    
    variables_classifier = current_metadata[:,[3,4,8,9,10,11,19,20,21,24,25,26,426,427,428,429,430,431,432,433,
                                               436,437,441,442,443,444,452,453,454,457,458,459,859,860,861,862,
                                               863,864,865,866,873,874]]
    y_val = model_reload.predict(variables_classifier) 
    keep_rows = np.where(y_val=='synaptic',True,False)
    y_val = y_val.reshape(len(y_val),1)
    
    current_tile_data = np.hstack((tile_ID,
                                   current_metadata[:,[7,6,440,439]],
                                   current_radii[:,[3,4,6,39]],
                                   non_neurite_intensity,
                                   current_metadata[:,[2,435,1]],
                                   current_metadata[:,[0]],
                                   current_metadata[:,[5,438]],
                                   y_val))
    
    
    
    total_data[i] = current_tile_data

set_loky_pickler('cloudpickle')    
Parallel(n_jobs=1, verbose=50, backend='threading')(delayed(get_data)(i)for i in range(0,np.shape(metadata_files)[0]))     


all_data = pd.DataFrame(np.vstack(total_data))
    
all_data.to_csv(out_folder  + out_file)

    
