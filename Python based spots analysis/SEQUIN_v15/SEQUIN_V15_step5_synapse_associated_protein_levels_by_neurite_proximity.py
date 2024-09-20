# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:02:15 2024

@author: andrew.sauerbeck
"""

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
###################################################################################################
## This code is specifically written to analyze synaptic centroids and a 200nm field around them.
###################################################################################################

###################################################################################################
## User defined required input parameters
###################################################################################################

####################################################
## Define input and output folders
####################################################
## input_folder_distances must be the same folder as the original SEQUIN_v15 output.
input_folder_distances = 'C:/SEQUIN_V15_output_folder/'
## Input folder is Output folder from code 'SEQUIN_V15_calculate_synaptic_density_around_masked_neurites'
input_folder = 'C:/SEQUIN_V15_calculate_synaptic_density_around_masked_neurites_output_folder/'
## Location and filename for image file.
input_czi_file = 'C:/image_file.czi'


####################################################
## Define background subtraction level
####################################################
## Manually calculate background subtraction value to apply.
third_channel_bls = 88


####################################################
## Define field radius size
####################################################
## Radius to sample around synaptic loci. Units of microns.
nm_cutoff = 0.2
## Maximum number of voxels for sampling. Must be greater than radii needed to nm_cutoff.
synaptic_field_voxel_radius = 52

####################################################
## Define channel alignment parameters
####################################################
## Precalculated channel alignment correction values.
shift_raw_ch0 = [2.16514106442566, 0.0, 0.0]
shift_raw_ch1 = [0, 0.0, 0.0]
shift_raw_ch2 = [-1.80336860507471, 0.0, 0.0]



#%%
##########################################################################################################
## Code block 1. Load imaging data and calculate synaptic protein levels by distance to masked neurites
## More steps in subsequent blocks
## Just run this block of code.
## Do not edit.
###########################################################################################################

files = np.sort(os.listdir(input_folder))
raw_image = AICSImage(input_czi_file,reconstruct_mosaic=False) 


czi_metadata = czi.CziFile(input_czi_file)

metadatadict_czi_True = czi_metadata.metadata(raw=True)
metadatadict_czi = czi_metadata.metadata(raw=False)


xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
voxelsize=[z_dim,xy_dim,xy_dim]




files_proximity = np.asarray([x for x in files if x.endswith('proximity_data.csv')] )   

files_data_by_loci = np.asarray([x for x in files if x.endswith('data_by_loci.csv')] )  



files_distances = np.sort(os.listdir(input_folder_distances))
files_radii = np.asarray([x for x in files_distances if x.endswith('3rd_channel_vRadii.csv')] )
files_distances = np.asarray([x for x in files_distances if x.endswith('neurite_positive_data.csv')] )

distance_file = files_distances[0]

distance_file_name =   (re.split('_MT_',distance_file))[0] + '_MT_'






first_file_proximity = np.asarray(dask.dataframe.read_csv(input_folder + files_proximity[0], delimiter=',', header=None)) 
first_file_by_loci = np.asarray(dask.dataframe.read_csv(input_folder + files_data_by_loci[0], delimiter=',', header=None)) 

compiled_data_proximity = np.hstack((np.asarray(['tile_id','tile_id','tile_id']).reshape(3,1),first_file_proximity[0:3,:]))
compiled_data_by_loci = first_file_by_loci[0,:]


current_file_distances = np.asarray(dask.dataframe.read_csv(input_folder_distances + files_distances[0], delimiter=',', header=None)) 
current_file_distances = current_file_distances[1::,1::]
current_file_distances_header = current_file_distances[0,:]



current_file_radii = np.asarray(dask.dataframe.read_csv(input_folder_distances + files_radii[0], delimiter=',', header=None)) 
current_file_radii = current_file_radii[1::,1::]

raw_synaptic_field_column = int(np.asarray(np.where(np.all(current_file_radii[0:5,:] == np.asarray(['raw','synaptic_centroid','52','0.2','mean']).reshape(5,1),axis=0))))
flipped_synaptic_field_column = int(np.asarray(np.where(np.all(current_file_radii[0:5,:] == np.asarray(['flipped','synaptic_centroid','52','0.2','mean']).reshape(5,1),axis=0))))
current_file_radii_header = current_file_radii[4,[0,1,2,3,4]]
current_file_radii_header = np.hstack((current_file_radii_header,'raw_synaptic_field','flipped_synaptic_field','mean_with_bls','median_with_bls'))

compiled_data_by_loci = np.hstack((compiled_data_by_loci,current_file_distances_header,'centroid_z','centroid_y','centroid_x',current_file_radii_header))


all_data = [None] * len(files_proximity)
#%
for file_i in range(0,len(files_proximity)):
    
    #%
    current_file_proximity = np.asarray(dask.dataframe.read_csv(input_folder + files_proximity[file_i], delimiter=',', header=None)) 
    current_file_proximity = current_file_proximity[3::,:]
    tile_ID = (re.search('tile_' + '(.*?)' + '_neurite', files_proximity[file_i]))
    tile_ID_repeat = np.repeat(tile_ID.group(1),np.shape(current_file_proximity)[0]).reshape(np.shape(current_file_proximity)[0],1)
    current_file_proximity = np.hstack((tile_ID_repeat,current_file_proximity))
    compiled_data_proximity = np.vstack((compiled_data_proximity,current_file_proximity))
    
    current_file_by_loci = np.asarray(dask.dataframe.read_csv(input_folder + files_data_by_loci[file_i], delimiter=',', header=None)) 
    current_file_by_loci = current_file_by_loci[1::,:].astype(float)
    
    
    
    current_file_distances = np.asarray(dask.dataframe.read_csv(input_folder_distances + distance_file_name + tile_ID.group(1) + '_neurite_positive_data.csv', delimiter=',', header=None)) 
    current_file_distances = current_file_distances[1::,1::]
    current_file_distances_header = current_file_distances[0,:]
    current_file_distances = (current_file_distances[1::,:]).astype(float)
    
    centroids_z = ((current_file_distances[:,2] + current_file_distances[:,10]) / 2).reshape(np.shape(current_file_distances)[0],1)
    centroids_y = ((current_file_distances[:,3] + current_file_distances[:,11]) / 2).reshape(np.shape(current_file_distances)[0],1)
    centroids_x = ((current_file_distances[:,4] + current_file_distances[:,12]) / 2).reshape(np.shape(current_file_distances)[0],1)
    
    centroids = np.hstack((centroids_z,centroids_y,centroids_x))
    centroids = np.round(centroids)
    
    current_file_radii = np.asarray(dask.dataframe.read_csv(input_folder_distances + distance_file_name + tile_ID.group(1) + '_3rd_channel_vRadii.csv', delimiter=',', header=None)) 
    current_file_radii = current_file_radii[1::,1::]
    
    current_file_radii = current_file_radii[5::,[0,1,2,3,4,raw_synaptic_field_column,flipped_synaptic_field_column]].astype(float)
    
    nn_test = np.sum(np.abs(np.round(current_file_distances[:,7],6) - np.round(current_file_radii[:,2],6)))
    
    if nn_test != 0:
        print('nn doesnt match, dont compile')
    if nn_test == 0:
        current_file_distances = np.hstack((current_file_distances,centroids,current_file_radii))
   ###
        current_image = raw_image.get_image_dask_data("ZYX", M=int(tile_ID.group(1)),C=0)  # returns out-of-memory 4D dask array
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
                
                
        image3 = np.where(image3 > third_channel_bls,image3 - third_channel_bls,0 )
        image_dims = np.shape(image3)
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

        #%
        keep = pairs[X<=radius]
        keep_nm = pairs[X_nm<=nm_cutoff]
        distance_data_sorted = []
        for row in current_file_by_loci:
            current_distance = current_file_distances[current_file_distances[:,18]==row[2]]
            current_distance = current_distance[current_distance[:,19]==row[3]]
            current_distance = current_distance[current_distance[:,20]==row[4]]
            current_distance = current_distance[0,:].reshape(1,28)
            
            
            
            current_centroid = current_distance[0,18:21].reshape(1,3)
            
            current_synaptic_loci_centroid_int = np.round(current_centroid.astype(float)).astype(np.int16)
            
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
            
            current_distance = np.hstack((current_distance,np.mean(third_channel_voxel_intensity).reshape(1,1),np.median(third_channel_voxel_intensity).reshape(1,1)))
            
            distance_data_sorted.append(current_distance)
            
            
        distance_data_sorted = np.vstack(distance_data_sorted)
        current_file_by_loci = np.hstack((current_file_by_loci,distance_data_sorted))                             
                                         
        print(file_i)
        all_data[file_i] = current_file_by_loci
        
#%       
        
        
        
        
all_data = np.vstack(all_data)


all_data = np.vstack((compiled_data_by_loci.reshape(1,len(compiled_data_by_loci)),all_data))
all_data_header = all_data[0,:]
all_data = all_data[1::,:].astype(float)

all_data = pd.DataFrame(all_data)      
all_data.to_csv(input_folder_distances  + 'compiled_neurite_data.csv',header=False,index=False) 


## Synaptic loci that do not have a detected neurite always have a detected distance ==0.01 more than max radius. 
## Impossible scenario to actually happen but allows for dectection of synaptic loci without detected neurites.
neurite_positive_test = all_data[:,39] - all_data[:,40]
all_data_subset = all_data[neurite_positive_test<=0,:]

#%% 
####################################################################
## Optional block of code for binning data by user defined intervals
#####################################################################
## Column 39 is the distance a synaptic loci is from a labeled neurite
## Column 43 is the average third_channel intensity in the field around that synaptic loci
all_data_subset_intensities = all_data_subset[:,[39,43]]
## Bin start in microns
bin_start = 0
## Bin interval in microns
bin_interval = 0.1
## Number of Bin intervals to create
number_of_bins = 21
current_image_data = []
for bin_i in range (1,number_of_bins):
    bin_end = bin_start + bin_interval
    current_subset = all_data_subset_intensities[all_data_subset_intensities[:,0].astype(float)>bin_start]
    current_subset = current_subset[current_subset[:,0].astype(float)<bin_end]
    current_subset = current_subset[:,1]
    current_subset_mean = np.mean(current_subset)
    current_subset_median = np.median(current_subset)
    current_subset_25th = np.quantile(current_subset,q=.25)
    current_subset_75th = np.quantile(current_subset,q=.75)
    
    
    
    
    current_image_data.append(np.hstack((np.mean([bin_start,bin_end]),current_subset_mean,current_subset_median,current_subset_25th,current_subset_75th)))
    
    
    bin_start  = bin_start  + 0.1

current_image_data = np.vstack(current_image_data)
 