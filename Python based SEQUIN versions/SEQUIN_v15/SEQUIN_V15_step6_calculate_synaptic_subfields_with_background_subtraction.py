# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:20:23 2024

@author: andrew.sauerbeck
"""
#########################################
## Load required packages
#########################################

import dask.dataframe
import pandas  as pd
import numpy as np
import skimage.feature as feature
import skimage.filters as filters
import time
import czifile as czi
import skimage.measure as measure
import skimage.segmentation as seg
import skimage.morphology as morph
import os
import re
import multiprocessing as mp
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import wrap_non_picklable_objects
from sys import getsizeof
import scipy
import tifffile
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy import exp
from scipy import ndimage as ndi
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util,restoration,transform
)
from aicsimageio import AICSImage
import gc
#%%
#########################################
## User defined required input parameters
#########################################

####################################################
## Define input and output folders
####################################################
## Master compiled date from LightGBM filtering
input_file = 'C:/SEQUIN_V15_step2_output.csv'
## Output folder. Must be manually created and exist before running code.
output_folder = 'C:/output_folder/'
## Location and filename for image file.
input_czi_file = 'C:?image_file.czi'
## Maximum Nearest Neighbor cutoff for synaptic loci in microns.
max_nn_cutoff = 0.5

####################################################
## Define channel and background subtraction level
####################################################
## Channel to analyze intensity data around synaptic centroids.
## Channel corresponds to original imaging data.
channel_to_analyze = 0
## Manually calculate background subtraction value to apply.
third_channel_bls = 88


####################################################
## Define field radius size
####################################################
## Radius to sample around synaptic loci. Units of microns.
nm_cutoff = 0.2
## Maximum number of voxels for sampling. Must be greater than radii needed to nm_cutoff.
synaptic_field_voxel_radius = 52


######################################################################################
## Third channel masking for synaptic loci distance measurements
## All masking based on third channel defined above
######################################################################################
## Intensity cutoff for third channel masking. Must be manually calculated.
neurite_mask_intensity_cutoff = 110
## Quantile intensity cutoff for third channel masking. Must be manually calculated.
neurite_mask_gauss_quantile_cutoff = .0015


####################################################
## Define channel alignment parameters
####################################################

## Precalculated channel alignment correction values.
shift_raw_ch0 = [2.16514106442566, 0.0, 0.0]
shift_raw_ch1 = [0, 0.0, 0.0]
shift_raw_ch2 = [-1.80336860507471, 0.0, 0.0]

####################################################
## Define location and filename for Tile ID ket
####################################################
## Location and filename for key for loading tileIDs to process
## Manually create based on Step 3f
tile_id_key = 'C:/tile_ids_file.csv'


#%%
###############################################################
## Code block 1. Load imaging data and reprocess fields around synaptic loci.
## More steps in subsequent blocks
## Just run this block of code.
## Do not edit.
##############################################################

tile_id_key = np.asarray(dask.dataframe.read_csv(tile_id_key, delimiter=',', header=None)) 
tile_id_key_header = tile_id_key[0,:]
tile_id_key = tile_id_key[1::,:]

input_file_data_all = np.asarray(dask.dataframe.read_csv(input_file, delimiter=',', header=None)) 
input_file_data_all = input_file_data_all[1::,1::]
input_file_data_all = input_file_data_all[input_file_data_all[:,12] < max_nn_cutoff]


#%
unique_tiles = np.unique(input_file_data_all[:,0])
raw_image = AICSImage(input_czi_file,reconstruct_mosaic=False) 

#%
czi_metadata = czi.CziFile(input_czi_file)

# parse the XML into a dictionary

#metadatadict_czi = czi.metadata(raw=False)
metadatadict_czi_True = czi_metadata.metadata(raw=True)
metadatadict_czi = czi_metadata.metadata(raw=False)


xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
voxelsize=[z_dim,xy_dim,xy_dim]




def process_synaptic_subfields(i):
    #%
    input_file_data_current = input_file_data_all[input_file_data_all[:,0] == i]
    input_file_data_current = input_file_data_current[input_file_data_current[:,16] == 'synaptic']
    current_image = raw_image.get_image_dask_data("ZYX", M=i,C=channel_to_analyze)  # returns out-of-memory 4D dask array
    image3 = current_image.compute()
    
    image3 = scipy.ndimage.shift(image3, shift=shift_raw_ch0,
                                   order=3, mode='constant', cval=0.0, prefilter=False)


    all_shifts = [shift_raw_ch0,shift_raw_ch1,shift_raw_ch2]
    for current_shift in all_shifts:
        shift = np.asarray(current_shift)
        shift = abs(shift.astype(int))
        edge_crop = shift + 1 
        image_shape_crop = np.shape(image3)
        #Z crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.
        if current_shift[0] > 0:
          
            image3 = image3[edge_crop[0]::,:,:]
        elif current_shift[0] < 0:
          
            image3 = image3[0:(image_shape_crop[0]-edge_crop[0]),:,:]
            
        #Y crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
        if current_shift[1] > 0:
          
            image3 = image3[:,edge_crop[1]::,:]
        elif current_shift[1] < 0:
         
            image3 = image3[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
            
            
        #X crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
        if current_shift[2] > 0:
        
            image3 = image3[:,:,edge_crop[2]::]
        elif current_shift[2] < 0:
           
            image3 = image3[:,:,0:(image_shape_crop[2]-edge_crop[2])] 
    
    scaling_factor = voxelsize[0] / voxelsize[1]
    low_gauss = filters.gaussian(image3,sigma=[1,np.round(scaling_factor),np.round(scaling_factor)])
    high_gauss = filters.gaussian(image3,sigma=[2,2*np.round(scaling_factor),2*np.round(scaling_factor)])
    filtered = (low_gauss - high_gauss)
    filtered_cutoff = np.where(filtered > neurite_mask_gauss_quantile_cutoff ,1,0)
   
    ## Merge mask with raw intensity
    is_this_real = filtered_cutoff * image3
    
    ##Select mask above cutoff
    neurite_mask = np.where(is_this_real > neurite_mask_intensity_cutoff,255,0)
   
    synaptic_centroids = np.asarray([(input_file_data_current[:,14]+input_file_data_current[:,15])/2,
                                     (input_file_data_current[:,2]+input_file_data_current[:,4])/2,
                                     (input_file_data_current[:,1]+input_file_data_current[:,3])/2]).T
    synaptic_centroids = np.round(synaptic_centroids.astype(float)).astype(np.uint16)
  
   
    image_dims = np.shape(image3)
    parenchymal_voxels = image3[neurite_mask == 0]
    parenchymal_voxels = np.where(parenchymal_voxels > third_channel_bls,parenchymal_voxels - third_channel_bls,0)
    
    parenchymal_voxels_mean = np.mean(parenchymal_voxels)
    parenchymal_voxels_median = np.median(parenchymal_voxels)
    image3 = np.where(image3 > third_channel_bls,image3 - third_channel_bls,0 )
    intensity_image_3rd_channel_flipped = np.flip(image3)
    xy_radius_scale = 1
    radius = synaptic_field_voxel_radius                                                            
    radius_check = synaptic_field_voxel_radius ##If manuall switch to radius on line above
    coords = np.linspace(-(radius * xy_radius_scale),(radius * xy_radius_scale),num=(((radius * xy_radius_scale)*2)+1),dtype=int)
    coords = coords.astype(np.int16)
    pairs = np.asarray(np.meshgrid(coords,coords,coords)).T.reshape(-1,3)
    pairs_nm = pairs * voxelsize
    X= np.sqrt((pairs[:,0]**2)+(pairs[:,1]**2)+(pairs[:,2]**2))
    X_nm= np.sqrt((pairs_nm[:,0]**2)+(pairs_nm[:,1]**2)+(pairs_nm[:,2]**2))
    #Select only spots within a max distance in pixels from centroid
    keep = pairs[X<=radius]
    keep_nm = pairs[X_nm<=nm_cutoff]
    
    

    all_3rd_channel_intensity_data = []
    all_3rd_channel_intensity_data_flipped = []
    
    
    current_synaptic_loci_centroid = synaptic_centroids[0]
    for row_i in range(0,np.shape(synaptic_centroids)[0]):
        row = synaptic_centroids[row_i]
        current_synaptic_loci_centroid_int = np.round(row.astype(float)).astype(np.int16)
        
        current_synaptic_field = current_synaptic_loci_centroid_int + keep_nm
        current_synaptic_field_org = current_synaptic_field
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,0] >0 ]
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,0] <(image_dims[0]-1) ]
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,1] >0 ]
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,1] <(image_dims[1]-1) ]
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,2] >0 ]
        current_synaptic_field = current_synaptic_field[current_synaptic_field[:,2] <(image_dims[2]-1) ]
        number_synaptic_voxels = len(current_synaptic_field)
        
        ## safety to remove small fields and crashed processing
        ## can be changed to only process fully filled fields, though would causes issues with really large fields
        if number_synaptic_voxels > 3:


            z_indices = np.array([current_synaptic_field[:,0]], dtype=np.intp)
            y_indices = np.array([current_synaptic_field[:,1]], dtype=np.intp)
            x_indices = np.array([current_synaptic_field[:,2]], dtype=np.intp)
            zyx_indices = np.vstack((z_indices,y_indices,x_indices)).T
            third_channel_voxel_intensity = image3[z_indices,y_indices,x_indices]
            third_channel_voxel_intensity_flipped = intensity_image_3rd_channel_flipped[z_indices,y_indices,x_indices]
            third_channel_intensity_data = np.hstack((
                number_synaptic_voxels,
                np.mean(third_channel_voxel_intensity),
                np.median(third_channel_voxel_intensity),
                np.min(third_channel_voxel_intensity),
                np.max(third_channel_voxel_intensity),
                np.std(third_channel_voxel_intensity))).reshape(1,6)
            all_3rd_channel_intensity_data.append(third_channel_intensity_data)
            
            third_channel_intensity_data_flipped = np.hstack((
                number_synaptic_voxels,
                np.mean(third_channel_voxel_intensity_flipped),
                np.median(third_channel_voxel_intensity_flipped),
                np.min(third_channel_voxel_intensity_flipped),
                np.max(third_channel_voxel_intensity_flipped),
                np.std(third_channel_voxel_intensity_flipped))).reshape(1,6)
            all_3rd_channel_intensity_data_flipped.append(third_channel_intensity_data_flipped)          

  
    
    all_3rd_channel_intensity_data = np.vstack(all_3rd_channel_intensity_data)

    all_3rd_channel_intensity_data_flipped = np.vstack(all_3rd_channel_intensity_data_flipped)     
    
    
    output_header = ['tile_id','raw_#_voxels','raw_mean','raw_median','raw_min','raw_max','raw_std','flipped_#_voxels','flipped_mean','flipped_median','flipped_min','flipped_max','flipped_std','parenchymal_intensity_mean','parenchymal_intensity_median']
    
    tile_id = np.repeat(i,np.shape(all_3rd_channel_intensity_data)[0]).reshape(np.shape(all_3rd_channel_intensity_data)[0],1)
    parenchymal_voxels_mean = np.repeat(parenchymal_voxels_mean,np.shape(all_3rd_channel_intensity_data)[0]).reshape(np.shape(all_3rd_channel_intensity_data)[0],1)
    parenchymal_voxels_median = np.repeat(parenchymal_voxels_median,np.shape(all_3rd_channel_intensity_data)[0]).reshape(np.shape(all_3rd_channel_intensity_data)[0],1)
    
    total_data = pd.DataFrame(np.vstack((output_header,np.hstack((tile_id,all_3rd_channel_intensity_data,all_3rd_channel_intensity_data_flipped,parenchymal_voxels_mean,parenchymal_voxels_median)))))      
    total_data.to_csv(output_folder  + 'tile_' + str(i) + '_reprocessed_synaptic_fields.csv',header=False,index=False)     


set_loky_pickler('cloudpickle')    
Parallel(n_jobs=6, verbose=50, backend='loky')(delayed(process_synaptic_subfields)(i)for i in unique_tiles)      
#%%
###############################################################
## Code block 2. Load and combine prior processing saved by file
## Synaptic loci screened to be in user specific tiles.
## Only change 'output_folder' if picking up from previous run of code.
##############################################################

####################################################################
## Define column for TileIDs and, if needed, the prior output_folder
####################################################################
#output_folder = 'C:/output_folder/'
## Column in tile_id_key the corresponds to currnt tilescan to process
column_i = 5

files = np.sort(os.listdir(output_folder))
files = np.asarray([x for x in files if x.endswith('reprocessed_synaptic_fields.csv')] )   


first_file = np.asarray(dask.dataframe.read_csv(output_folder + files[0], delimiter=',', header=None)) 
compiled_data = first_file[0,:]
for file in files:
    current_file = np.asarray(dask.dataframe.read_csv(output_folder + file, delimiter=',', header=None)) 
    current_file = current_file[1::,:].astype(float)
    current_file = np.median(current_file,axis=0)
    compiled_data = np.vstack((compiled_data,current_file))
compiled_data_header = compiled_data[0,:]      
compiled_data = compiled_data[1::,:]          
                  

current_image_column = tile_id_key[:,column_i]
current_tiles  = np.unique(current_image_column)
#%
current_subset_rows = np.isin(compiled_data[:,0],current_tiles)
current_subset_data = compiled_data[current_subset_rows,:]

