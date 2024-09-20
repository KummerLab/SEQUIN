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
## Maximum Nearest Neighbor cutoff for synaptic loci in microns.
max_nn_cutoff = 0.5
## Output folder. Must be manually created and exist before running code.
output_folder = 'C:/output_folder/'
## Location and filename for image file.
input_czi_file = 'C:/image_file.czi'

####################################################
## Define neurite masking parameters. 
####################################################
## Intensity cutoff for third channel masking. Must be manually calculated.
neurite_mask_intensity_cutoff = 110
## Quantile intensity cutoff for third channel masking. Must be manually calculated.
neurite_mask_gauss_quantile_cutoff = .0015

####################################################
## Define channel alignment values
####################################################
## Precalculated channel alignment correction values.
shift_raw_ch0 = [2.16514106442566, 0.0, 0.0]
shift_raw_ch1 = [0, 0.0, 0.0]
shift_raw_ch2 = [-1.80336860507471, 0.0, 0.0]

####################################################
## Define radius in microns to dilate neurite by
####################################################
## Dilation around neurites to test
radii_to_test = [0,0.2,0.4,0.8]

####################################################
## Set number of simultaneous processes
####################################################
number_of_simultaneous_jobs = 5

#%%
###############################################################
## Code block 1. Load imaging data and reprocess synaptic loci around masked neurites.
## More steps in subsequent blocks
## Just run this block of code.
## Do not edit.
##############################################################


#%

input_file_data_all = np.asarray(dask.dataframe.read_csv(input_file, delimiter=',', header=None)) 
input_file_data_all = input_file_data_all[1::,1::]
input_file_data_all = input_file_data_all[input_file_data_all[:,12] < max_nn_cutoff]

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





def synaptic_density_around_neurites(i):
    #%
    input_file_data_current = input_file_data_all[input_file_data_all[:,0] == i]
    current_image = raw_image.get_image_dask_data("ZYX", M=i,C=0)  # returns out-of-memory 4D dask array
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
    
    above_neurite_cutoff = np.where(is_this_real > neurite_mask_intensity_cutoff,1,0)
    above_neurite_cutoff_bool = above_neurite_cutoff.astype(bool)
    
    image_dims= np.shape(image3)
    all_neurite_distances = []
    all_neurite_radii  = []
    synaptic_centroids = np.asarray([(input_file_data_current[:,14]+input_file_data_current[:,15])/2,
                                     (input_file_data_current[:,2]+input_file_data_current[:,4])/2,
                                     (input_file_data_current[:,1]+input_file_data_current[:,3])/2]).T
    centroids = np.round(synaptic_centroids.astype(float)).astype(np.uint16)
    
        ##analysis across varying radii
    labeled_mask = measure.label(above_neurite_cutoff_bool)   
    unique_neurites = np.unique(labeled_mask)
    unique_neurites = unique_neurites[unique_neurites != 0]
    if (len(unique_neurites) == 0):
        total_data = pd.DataFrame(np.zeros((1,14)) )
        total_data.to_csv(output_folder  + 'tile_' + str(i) + '_no_neurites_to_analyze.csv',header=False,index=False)   
      

    data_by_loci = ['tile id', 'cutoff','z' , 'y', 'x', 'raw intensity', 'zeroed intensity',
                    'raw mean','raw area','raw counts','raw density',
                    'zeroed mean','zeroed area','zeroed counts','zeroed density']
    if (len(unique_neurites) > 0):
        above_neurite_cutoff_bool_invert = np.invert(above_neurite_cutoff_bool)
        for nm_cutoff_neurite in radii_to_test:    
            ellipse_radius = (int(np.round(nm_cutoff_neurite / voxelsize[1])))
            ellipse_array = np.zeros(((ellipse_radius*2)+1,(ellipse_radius*2)+1,(ellipse_radius*2)+1))
            center = [ellipse_radius,ellipse_radius,ellipse_radius]
            ellipse_xy_radius_scale = 1
            
            coords = np.linspace(-(ellipse_radius * ellipse_xy_radius_scale),(ellipse_radius * ellipse_xy_radius_scale),num=(((ellipse_radius * ellipse_xy_radius_scale)*2)+1),dtype=int)
            coords = coords.astype(np.int16)
            pairs = np.asarray(np.meshgrid(coords,coords,coords)).T.reshape(-1,3)
            pairs_nm = pairs * voxelsize
            X= np.sqrt((pairs[:,0]**2)+(pairs[:,1]**2)+(pairs[:,2]**2))
            X_nm= np.sqrt((pairs_nm[:,0]**2)+(pairs_nm[:,1]**2)+(pairs_nm[:,2]**2))
            
            keep_nm = pairs[X_nm<=nm_cutoff_neurite]
            current_field = center + keep_nm
            
            ellipse_array[current_field[:,0],current_field[:,1],current_field[:,2]] = 1
            
            
                
        
            dilated_mask = morphology.dilation(labeled_mask,footprint=ellipse_array)
            dilated_mask = dilated_mask.astype(np.uint16)
            
            dilated_mask_zeroed = dilated_mask * above_neurite_cutoff_bool_invert
                
            
            props = ('label','mean_intensity','min_intensity','max_intensity','area')
            neurite_data_current_raddi = measure.regionprops_table(label_image=dilated_mask, intensity_image=image3,properties=props)
            neurite_data_current_raddi = np.asarray(pd.DataFrame.from_dict(neurite_data_current_raddi, orient='columns'))
            
            neurite_data_current_raddi_zeroed = measure.regionprops_table(label_image=dilated_mask_zeroed, intensity_image=image3,properties=props)
            neurite_data_current_raddi_zeroed = np.asarray(pd.DataFrame.from_dict(neurite_data_current_raddi_zeroed, orient='columns'))
            
              
            neurite_positive = dilated_mask[centroids[:,0],centroids[:,1],centroids[:,2]]
            neurite_positive_zeroed = dilated_mask_zeroed[centroids[:,0],centroids[:,1],centroids[:,2]]
            current_cutcoff = np.repeat(nm_cutoff_neurite,np.shape(centroids)[0]).reshape(np.shape(centroids)[0],1)
            neurite_intensities = []
            neurite_intensities_zeroed = []
            for positive in  neurite_positive:
                if positive == 0:
                    neurite_intensities.append(0)
                if positive != 0:
                    current_neurite_intensity = neurite_data_current_raddi[neurite_data_current_raddi[:,0]==positive,1]
                    neurite_intensities.append(current_neurite_intensity)
            for positive in  neurite_positive_zeroed:
                if positive == 0:
                    neurite_intensities_zeroed.append(0)
                if positive != 0:
                    current_neurite_intensity = neurite_data_current_raddi_zeroed[neurite_data_current_raddi_zeroed[:,0]==positive,1]
                    neurite_intensities_zeroed.append(current_neurite_intensity)   
            neurite_intensities = np.vstack(neurite_intensities)
            neurite_intensities_zeroed = np.vstack(neurite_intensities_zeroed)
            current_tile_i_repeat = np.repeat(i,np.shape(centroids)[0]).reshape(np.shape(centroids)[0],1)
            
            #%
            loci_per_neurite = []
            for unique_neurite in unique_neurites:
                current_neurite_counts = len(neurite_positive[neurite_positive == unique_neurite])
                current_neurite_counts_zeroed = len(neurite_positive_zeroed[neurite_positive_zeroed == unique_neurite])
                loci_per_neurite.append(np.hstack((nm_cutoff_neurite,unique_neurite,current_neurite_counts,current_neurite_counts_zeroed)))
            loci_per_neurite = np.vstack(loci_per_neurite)
            #%
            if nm_cutoff_neurite == 0:
                neurite_ids_order_raw = neurite_data_current_raddi[:,0]
                neurite_ids_order_zeroed = neurite_data_current_raddi_zeroed[:,0]
                neurite_ids_test_raw = np.sum(np.abs(unique_neurites - neurite_ids_order_raw))
                total_data = np.hstack(('dilation_radius','mask_type','neurite_id',unique_neurites))
                total_data = total_data.reshape(len(total_data),1)
                if (neurite_ids_test_raw != 0):
                    print('ids dont match')
                    print('quitting')
                    
                elif (neurite_ids_test_raw == 0):
                    raw_data = neurite_data_current_raddi[:,[1,4]]
                    raw_data = np.hstack((raw_data,loci_per_neurite[:,2].reshape(np.shape(loci_per_neurite)[0],1)))
                    raw_density = (raw_data[:,2] / raw_data[:,0]).reshape(np.shape(loci_per_neurite)[0],1)
                    raw_data = np.hstack((raw_data,raw_density))
                    
                    neurite_densities = []
                    neurite_densities_zeroed = []
                    for positive in  neurite_positive:
                        if positive == 0:
                            neurite_densities.append(np.zeros((1,4)))
                        if positive != 0:
                            current_neurite_density = raw_data[unique_neurites==positive,:]
                            neurite_densities.append(current_neurite_density)
                   
                    neurite_densities = np.vstack(neurite_densities)
                    neurite_densities_zeroed = np.zeros((np.shape(neurite_densities)))
                    current_data = np.hstack((current_tile_i_repeat,current_cutcoff,centroids,neurite_intensities,neurite_intensities_zeroed,
                                              neurite_densities,neurite_densities_zeroed))
                    
                    data_by_loci = np.vstack((data_by_loci, current_data))
                    
                    raw_data_header = [[nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite],['raw','raw','raw','raw'],['mean','area','counts','density']]
                    raw_data = np.vstack((raw_data_header,raw_data))
                    
                    total_data = np.hstack((total_data,raw_data))
                    
            if nm_cutoff_neurite != 0:
                neurite_ids_order_raw = neurite_data_current_raddi[:,0]
                neurite_ids_order_zeroed = neurite_data_current_raddi_zeroed[:,0]
                neurite_ids_test_raw = np.sum(np.abs(unique_neurites - neurite_ids_order_raw))
                neurite_ids_test_zeroed = np.sum(np.abs(unique_neurites - neurite_ids_order_zeroed))
                if (neurite_ids_test_raw != 0) or (neurite_ids_test_zeroed != 0):
                    print('ids dont match')
                    print('quitting')
                    #return
                elif (neurite_ids_test_raw == 0) and (neurite_ids_test_zeroed == 0):
                    raw_data = neurite_data_current_raddi[:,[1,4]]
                    raw_data = np.hstack((raw_data,loci_per_neurite[:,2].reshape(np.shape(loci_per_neurite)[0],1)))
                    raw_density = (raw_data[:,2] / raw_data[:,0]).reshape(np.shape(loci_per_neurite)[0],1)
                    raw_data = np.hstack((raw_data,raw_density))
                    
                    zeroed_data = neurite_data_current_raddi_zeroed[:,[1,4]]
                    zeroed_data = np.hstack((zeroed_data,loci_per_neurite[:,3].reshape(np.shape(loci_per_neurite)[0],1)))
                    zeroed_density = (zeroed_data[:,2] / zeroed_data[:,0]).reshape(np.shape(loci_per_neurite)[0],1)
                    zeroed_data = np.hstack((zeroed_data,zeroed_density))
                    
                    
                    neurite_densities = []
                    neurite_densities_zeroed = []
                    for positive in  neurite_positive:
                        if positive == 0:
                            neurite_densities.append(np.zeros((1,4)))
                        if positive != 0:
                            current_neurite_density = raw_data[unique_neurites==positive,:]
                            neurite_densities.append(current_neurite_density)
                    for positive in  neurite_positive_zeroed:
                        if positive == 0:
                            neurite_densities_zeroed.append(np.zeros((1,4)))
                        if positive != 0:
                            current_neurite_density = zeroed_data[unique_neurites==positive,:]
                            neurite_densities_zeroed.append(current_neurite_density)   
                    neurite_densities = np.vstack(neurite_densities)
                    neurite_densities_zeroed = np.vstack(neurite_densities_zeroed)
                    current_data = np.hstack((current_tile_i_repeat,current_cutcoff,centroids,neurite_intensities,neurite_intensities_zeroed,
                                              neurite_densities,neurite_densities_zeroed))
                    
                    data_by_loci = np.vstack((data_by_loci, current_data))
                    
                    raw_data_header = [[nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite],['raw','raw','raw','raw'],['mean','area','counts','density']]
                    raw_data = np.vstack((raw_data_header,raw_data))
                    
                    zeroed_data_header = [[nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite,nm_cutoff_neurite],['zeroed','zeroed','zeroed','zeroed'],['mean','area','counts','density']]
                    zeroed_data = np.vstack((zeroed_data_header,zeroed_data))
                    total_data = np.hstack((total_data,raw_data,zeroed_data))

#

        data_by_loci = pd.DataFrame(data_by_loci)
        data_by_loci.to_csv(output_folder  + 'tile_' + str(i) + '_data_by_loci.csv',header=False,index=False)    


        total_data = pd.DataFrame(total_data)      
        total_data.to_csv(output_folder  + 'tile_' + str(i) + '_neurite_proximity_data.csv',header=False,index=False)     


set_loky_pickler('cloudpickle')    
Parallel(n_jobs=number_of_simultaneous_jobs, verbose=50, backend='loky')(delayed(synaptic_density_around_neurites)(i)for i in unique_tiles)      

#%%
###############################################################
## Code block 2. Load and combine prior processing saved by file
## Synaptic loci screened to be in user specific tiles.
## Only changes needed are if 'output_folder' needs to be defined to pick up from previous run of code.
##############################################################
## Input folder automatically set to prior output folder
input_folder = output_folder

#input_folder = 'C:/output_folder/'


files = np.sort(os.listdir(input_folder))
files = np.asarray([x for x in files if x.endswith('proximity_data.csv')] )   


first_file = np.asarray(dask.dataframe.read_csv(input_folder + files[0], delimiter=',', header=None)) 

compiled_data = np.hstack((np.asarray(['tile_id','tile_id','tile_id']).reshape(3,1),first_file[0:3,:]))
for file in files:
    current_file = np.asarray(dask.dataframe.read_csv(input_folder + file, delimiter=',', header=None)) 
    current_file = current_file[3::,:]
    tile_ID = (re.search('tile_' + '(.*?)' + '_neurite', file))
    tile_ID = np.repeat(tile_ID.group(1),np.shape(current_file)[0]).reshape(np.shape(current_file)[0],1)
    current_file = np.hstack((tile_ID,current_file))
    compiled_data = np.vstack((compiled_data,current_file))

#%%
##################################################################
## Code block 3. Define column to process and interval to bin data
#################################################################

## Column in Block 2 below to analyze synaptic density. This wll be dependant on radii_to_test above and experimental question.
compiled_data_column_to_analyze = 28
## Intensity interval to bin data by
interval = 1000


data_set = compiled_data[3::,:]
data_set = data_set[data_set[:,0].astype(float) >0]

bin_start = 0
all_data = []
for bin_i in range(1,21):
    bin_end = bin_i * interval
    current_bin = data_set[data_set[:,2].astype(float)>bin_start]
    current_bin = current_bin[current_bin[:,2].astype(float)<bin_end]
    current_bin = current_bin[current_bin[:,compiled_data_column_to_analyze].astype(float)>=1]    
    all_data.append(np.hstack((bin_start,bin_end,np.median(current_bin[:,2].astype(float)),np.median(current_bin[:,29].astype(float)))))
    bin_start = bin_end
all_data = np.vstack(all_data)   

