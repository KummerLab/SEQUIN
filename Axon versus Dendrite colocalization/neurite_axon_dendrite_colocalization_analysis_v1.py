# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:18:31 2026

@author: andrew.sauerbeck
"""
#########################################
## Load required packages
#########################################

import dask.dataframe
import pandas  as pd
import numpy as np
import czifile as czi
import skimage.morphology as morph
import os
import re
from sys import getsizeof
import tifffile
import pickle
import matplotlib.pyplot as plt
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util,restoration,transform
)
from aicsimageio import AICSImage
import gc
from joblib import dump, load
import scipy
#%%
#########################################
## User defined required input parameters
#########################################

####################################################
## Define input and output folders
####################################################
file ='G:/raw_image.czi'
outfolder = 'G:/location_output_images/'
temp_output_folder = 'G:/image_saves_for_rerun/'
save_name = 'section1'

####################################################
## Define location of masks for colocalization
## User must determine parameters for generating
## Binary masks used for colocalization
## Background must be set to 0
## Masked area must be set to 1
####################################################
somal_mask = 'G:/somal_mask.tif'
map2_mask = 'G:/map2_mask.tif'
nfh_mask = 'G:/nfh_mask.tif'

####################################################
## Set neurite mask intensity cutoff
## Must be manually calculated before masking
## Recommended value is 95th percentile
####################################################
neurite_mask_cutoff = 105

##################################
## Channel Alignment of raw images
##################################
## User defined channel alignment corrections for each channel
shift_raw_ch0 = [-1.48759575906035, 0, 0]
shift_raw_ch1 = [0.475861022818279, 0, 0]
shift_raw_ch2 = [0, 0, 0]

##################################
## Colocalization parameters
##################################
mask_type = 'negative'
selection_type = 'percent'
psyn_size_max = 1500000

##################################
## Suffix for output image
##################################
suffix = 'Z11-22'


################################################
## User must confirm the tiling key correction 
## for the size of the Airyscan processed image
## versus raw to adjust the mosaic tiling key
###############################################
tile_key_correction = 1142


############################################################
## User must confirm the Z dimension of the 3D images
## after correction for chromatic aberration has been applied
############################################################
number_z_slices = 67

#%%
##################################
## Load definitions and build
## mosaic tiling keys
## Contains Advanced parameters
## Recommend leaving as defaults
##################################

####################################
## Default is no gaussian smoothing
####################################
gaussian_filter_image = False


######################################
## Zeiss LSM980 Airysacn image spacing
######################################
tile_zero_spacing=12


#######################################
## Intensity cutoff for mask refinement
#######################################
mask_intensity_cutoff = 0.05


def GetCZImetadata(current_row_file,scene):
    current_metadatadict_czi =(czi.CziFile(current_row_file)).metadata(raw=False)
    current_metadatadict_czi_table =(czi.CziFile(current_row_file)).metadata(raw=True)
    #current_metadatadict_czi = czi_metadata.metadata(raw=False)#(folder + '/' + file)
    #image_dims = np.shape(image)
    #Find current image tile name
    curent_image_tile_ID = re.split('-',current_row_file)
    curent_image_tile_ID = curent_image_tile_ID[-1]
    curent_image_tile_ID = curent_image_tile_ID[0:-4]

    #current_image_tile_number = int((np.where(tile_region_names==curent_image_tile_ID))[0])

    current_tile_columns = (current_metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion']
              [0]
              ['Columns'])
              
    current_tile_rows = (current_metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion']
              [0]
              ['Rows'])          

    current_tile_SizeX = (current_metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Information']
              ['Image']
              ['SizeX'])
              
    current_tile_SizeY = (current_metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Information']
              ['Image']
              ['SizeY'])
    


    return current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY,current_metadatadict_czi, current_metadatadict_czi_table



    
def z_intensity_correction(image_to_adjust):
    
    image_allz_median = np.median(image_to_adjust)
    image_z_median = []
    for z in range(0,len(image_to_adjust[:,1,1])):
        current_image_z_median = np.median(image_to_adjust[z,:,:])
        image_z_median.append(current_image_z_median)
        
    image_z_correction = image_allz_median / image_z_median
    
    image_z_corrected = np.zeros((np.shape(image_to_adjust)))
    for z in range(0,len(image_to_adjust[:,1,1])):
        image_z_corrected[z,:,:] = image_to_adjust[z,:,:] * image_z_correction[z]
    
    image_z_corrected = image_z_corrected.astype(np.uint16)
   
    return image_z_corrected

#%
scene = 0
scale=1
tile_zero_spacing=12

img = AICSImage(file,reconstruct_mosaic=False) 
czi_metadata = czi.CziFile(file)

metadatadict_czi_True = czi_metadata.metadata(raw=True)
metadatadict_czi = czi_metadata.metadata(raw=False)

xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
voxelsize=[z_dim,xy_dim,xy_dim]



current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY,current_metadatadict_czi, current_metadatadict_czi_table = GetCZImetadata(file,scene)



lazy_t0 = img.get_image_dask_data("ZYX", M=0,C=0)  # returns out-of-memory 4D dask array
t0 = lazy_t0.compute()
t0 = t0[int((np.shape(t0)[0])/2),:,:]
actual_Y_tile_size = int(np.shape(t0)[0] * scale)
actual_X_tile_size = int(np.shape(t0)[1] * scale)


### Build mosaic key

directory = czi_metadata.filtered_subblock_directory
mosaic_key = []
for directory_entry in directory:
    mosaic_index = directory_entry.mosaic_index
    mosaic_start = directory_entry.start
    mosaic_key.append(np.hstack((mosaic_index,mosaic_start)))
mosaic_key = np.vstack(mosaic_key)
mosaic_key = mosaic_key[mosaic_key[:,1]==0]
## Select slice 0
mosaic_key = mosaic_key[mosaic_key[:,4]==0]
## Select channel 0
mosaic_key = mosaic_key[mosaic_key[:,3]==0]


mosaic_key2 = mosaic_key.copy()
mosaic_key2[:,5] = mosaic_key2[:,5] / tile_key_correction * 12
mosaic_key2[:,6] = mosaic_key2[:,6] / tile_key_correction * 12

mosaic_key[:,5] = mosaic_key[:,5] / tile_key_correction * actual_Y_tile_size
mosaic_key[:,6] = mosaic_key[:,6] / tile_key_correction * actual_X_tile_size

mosaic_key2[:,5] = mosaic_key[:,5] + mosaic_key2[:,5]
mosaic_key2[:,6] = mosaic_key[:,6] + mosaic_key2[:,6]
#%%
################################################################
## Main analysis code
## Only need to run this portion of code.
## No parameters to adjust
## If prior processing already performed, user can skip this block
##################################################################




out_ch0 = np.zeros((int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_ch1 = np.zeros((int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_ch2 = np.zeros((int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_ch3 = np.zeros((int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_neurite_mask = np.zeros((int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)

out_ch0_3d = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_ch1_3d = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_ch2_3d = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)
out_neurite_mask_3d = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint16)#.astype(np.uint16)

#### Check if tile size matched metadata and image

current_tile_SizeX = czi_metadata.shape[5]
current_tile_SizeY = czi_metadata.shape[4]


actual_X_image_pixels = current_tile_SizeX - (tile_zero_spacing * (current_tile_columns-1))
actual_Y_image_pixels = current_tile_SizeY - (tile_zero_spacing * (current_tile_rows-1))


all_data = []
psyn_voxels_all = []
nfh_voxels_all = []
map2_voxels_all = []


psyn_voxels_all_max_cutoff = []
nfh_voxels_all_max_cutoff = []
map2_voxels_all_max_cutoff = []

for mosaic_tile_i in range(0,img.shape[0]):
    print(mosaic_tile_i)
    
    lazy_t0 = img.get_image_dask_data("ZYX", M=mosaic_tile_i,C=0)  # returns out-of-memory 4D dask array
    t0 = lazy_t0.compute()
    
    lazy_t1 = img.get_image_dask_data("ZYX", M=mosaic_tile_i,C=1)  # returns out-of-memory 4D dask array
    t1 = lazy_t1.compute()
    
    lazy_t2 = img.get_image_dask_data("ZYX", M=mosaic_tile_i,C=2)  # returns out-of-memory 4D dask array
    t2 = lazy_t2.compute()
   
    y_coord = int(mosaic_key[mosaic_tile_i,5])
    x_coord = int(mosaic_key[mosaic_tile_i,6])
    
    y_coord_3d = int(mosaic_key2[mosaic_tile_i,5])
    x_coord_3d = int(mosaic_key2[mosaic_tile_i,6])
   
    

    out_ch0[y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= np.max(t0,axis=0)
    out_ch1[y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= np.max(t1,axis=0)
    out_ch2[y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= np.max(t2,axis=0)
    
 



    t0 = scipy.ndimage.shift(t0, shift=shift_raw_ch0,
                                   order=3, mode='constant', cval=0.0, prefilter=False)


    t1 = scipy.ndimage.shift(t1, shift=shift_raw_ch1,
                                   order=3, mode='constant', cval=0.0, prefilter=False)


    t2 = scipy.ndimage.shift(t2, shift=shift_raw_ch2,
                                   order=3, mode='constant', cval=0.0, prefilter=False)
    if gaussian_filter_image == True:
        t0 = filters.gaussian(t0, sigma=gaussian)
        t1 = filters.gaussian(t1, sigma=gaussian)
        t2 = filters.gaussian(t2, sigma=gaussian)

    all_shifts = [shift_raw_ch0,shift_raw_ch1,shift_raw_ch2]
    for current_shift in all_shifts:
        shift = np.asarray(current_shift)
        shift = abs(shift.astype(int))
        edge_crop = shift + 1 
        image_shape_crop = np.shape(t0)
        #Z crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.
        if current_shift[0] > 0:
            t0 = t0[edge_crop[0]::,:,:]
            t1 = t1[edge_crop[0]::,:,:]
            t2 = t2[edge_crop[0]::,:,:]
           # image4 = image4[edge_crop[0]::,:,:]
        elif current_shift[0] < 0:
            t0 = t0[0:(image_shape_crop[0]-edge_crop[0]),:,:]
            t1 = t1[0:(image_shape_crop[0]-edge_crop[0]),:,:]
            t2 = t2[0:(image_shape_crop[0]-edge_crop[0]),:,:]
           # image4 = image4[0:(image_shape_crop[0]-edge_crop[0]),:,:]
            
        #Y crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
        if current_shift[1] > 0:
            t0 = t0[:,edge_crop[1]::,:]
            t1 = t1[:,edge_crop[1]::,:]
            t2 = t2[:,edge_crop[1]::,:]
           # image4 = image4[:,edge_crop[1]::,:]
        elif current_shift[1] < 0:
            t0 = t0[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
            t1 = t1[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
            t2 = t2[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
          #  image4 = image4[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
            
            
        #X crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
        if current_shift[2] > 0:
            t0 = t0[:,:,edge_crop[2]::]
            t1 = t1[:,:,edge_crop[2]::]
            t2 = t2[:,:,edge_crop[2]::]
           # image4 = image4[:,:,edge_crop[2]::]
        elif current_shift[2] < 0:
            t0 = t0[:,:,0:(image_shape_crop[2]-edge_crop[2])]
            t1 = t1[:,:,0:(image_shape_crop[2]-edge_crop[2])] 
            t2 = t2[:,:,0:(image_shape_crop[2]-edge_crop[2])] 
           #image4 = image4[:,:,0:(image_shape_crop[2]-edge_crop[2])] 

 
    
    t0 = z_intensity_correction(t0)
    t1 = z_intensity_correction(t1)
    t2 = z_intensity_correction(t2)
    
    
    out_ch0_3d[0:number_z_slices,y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= t0
    out_ch1_3d[0:number_z_slices,y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= t1
    out_ch2_3d[0:number_z_slices,y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= t2
  

    min_cutoff = np.quantile(t0,q=.95)
    neurite_mask0=np.where(t0>neurite_mask_cutoff,1,0)


    psyn_voxels = t0[neurite_mask0==1]
    nfh_voxels = t1[neurite_mask0==1]
    map2_voxels = t2[neurite_mask0==1]
    
    psyn_voxels_all.append(psyn_voxels)
    nfh_voxels_all.append(nfh_voxels)
    map2_voxels_all.append(map2_voxels)


    max_cutoff = np.quantile(psyn_voxels,q=.98)
    
    nfh_voxels = nfh_voxels[psyn_voxels < max_cutoff]
    map2_voxels = map2_voxels[psyn_voxels < max_cutoff]
    psyn_voxels = psyn_voxels[psyn_voxels < max_cutoff]
    
    psyn_voxels_all_max_cutoff.append(psyn_voxels)
    nfh_voxels_all_max_cutoff.append(nfh_voxels)
    map2_voxels_all_max_cutoff.append(map2_voxels)
     
    neurite_mask1= 1
    
    out_neurite_mask_3d[0:number_z_slices,y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]= (neurite_mask0*neurite_mask1)

    histogram, bin_edges = np.histogram(psyn_voxels[psyn_voxels<max_cutoff], bins=100)
    

    for row_i in range(0,len(bin_edges)-1):
        nfh_voxels_current = nfh_voxels[np.where((psyn_voxels>bin_edges[row_i]) & (psyn_voxels<bin_edges[row_i + 1]))]
        map2_voxels_current = map2_voxels[np.where((psyn_voxels>bin_edges[row_i]) & (psyn_voxels<bin_edges[row_i + 1]))]
        psyn_voxels_current =psyn_voxels[np.where((psyn_voxels>bin_edges[row_i]) & (psyn_voxels<bin_edges[row_i + 1]))]
        
        
        all_data.append(np.hstack((np.mean(psyn_voxels_current),np.mean(nfh_voxels_current),np.mean(map2_voxels_current),np.count_nonzero(psyn_voxels_current),mosaic_tile_i)))
                        
    
    
    
    out_neurite_mask[y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size]=np.max((neurite_mask0*neurite_mask1),axis=0)
  
    
all_data = np.vstack(all_data)


## Save prior 3d output for faster rerun 4/1/2026
dump(out_ch0_3d, temp_output_folder + save_name + '_out_ch0_3d.joblib')
dump(out_ch1_3d, temp_output_folder + save_name + '_out_ch1_3d.joblib')
dump(out_ch2_3d, temp_output_folder + save_name + '_out_ch2_3d.joblib')
dump(out_neurite_mask_3d, temp_output_folder + save_name + '_out_neurite_mask_3d.joblib')
#%%
########################################################
## If image loading and preprocessing already completed
## User can directly load the output from pior processing
########################################################


temp_output_folder = 'G:/asap/Brain manuscript 2026/s1 tile imaging/neurite_analysis/AS4/data_with_somal_masks/3d image saves for rerun/'
out_ch0_3d = load(temp_output_folder + save_name + '_out_ch0_3d.joblib')
out_ch1_3d = load(temp_output_folder + save_name + '_out_ch1_3d.joblib')
out_ch2_3d = load(temp_output_folder + save_name + '_out_ch2_3d.joblib')
out_neurite_mask_3d = load(temp_output_folder + save_name + '_out_neurite_mask_3d.joblib')
#%%
#########################################################
## Perform colocalization analysis and save output images
#########################################################



#######################################################
## User should manually choose what slices to use
## for maximum intensity projections in representative
## output images
#######################################################
projection_z_start = 11
prejection_z_end = 22


def output_max_projections(output_max_projections_image):
    output_max_projections_image = np.max(output_max_projections_image[projection_z_start:prejection_z_end,:,:,:],axis=0)
    return output_max_projections_image



#% load and resize filter somal mask
somal_mask_image = np.invert(tifffile.imread(somal_mask))
somal_mask_image_large = transform.resize(somal_mask_image, output_shape=[np.shape(out_ch0_3d)[0],int(np.shape(out_ch0_3d)[1]),int(np.shape(out_ch0_3d)[2])], preserve_range=True, anti_aliasing=False)

tile_id_map = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint8)#.astype(np.uint16)

border_crop = np.zeros((number_z_slices,int(actual_Y_tile_size * current_tile_rows * 1),int(actual_X_tile_size * current_tile_columns* 1)),dtype=np.uint8)#.astype(np.uint16)


map2_mask_image = (tifffile.imread(map2_mask)).astype(np.uint8)
nfh_mask_image = (tifffile.imread(nfh_mask)).astype(np.uint8)

for mosaic_tile_i in range(0,img.shape[0]):
     
    y_coord = int(mosaic_key[mosaic_tile_i,5])
    x_coord = int(mosaic_key[mosaic_tile_i,6])
    current_tile_id = int(mosaic_key[mosaic_tile_i,0])
    
    border_crop[5:number_z_slices-5,(y_coord + 12):(y_coord+actual_Y_tile_size-12),(x_coord+12):(x_coord+actual_X_tile_size-12)]= 1
    tile_id_map[0:number_z_slices,y_coord:y_coord+actual_Y_tile_size,x_coord:x_coord+actual_X_tile_size] = current_tile_id

map2_mask_image = map2_mask_image * border_crop
nfh_mask_image = nfh_mask_image * border_crop





all_areas_map2 = []
all_areas_nfh = []
seeds = []
keep = []


all_areas = []
seeds = []
keep = []

labeled_neurites_size_filters = out_neurite_mask_3d * border_crop * somal_mask_image_large


labeled_neurites2 = measure.label(labeled_neurites_size_filters)
properties = measure.regionprops(label_image=labeled_neurites2)# 

all_areas2 = []
seeds = []
keep2 = []
for row in properties:
    all_areas2.append(row.area)
    
    
    if row.area < psyn_size_max:
       if row.area > 10:
           keep2.append(row.label)
           all_areas2.append(row.area)
        
all_areas2=np.vstack(all_areas2)
keep2 = np.vstack(keep2)

labeled_neurites_size_filters2 = np.isin(labeled_neurites2,keep2)


    
props = ('label','area','mean_intensity')
labeled_neurites_size_filters2 = measure.label(labeled_neurites_size_filters2)
if np.max(labeled_neurites_size_filters2) < 65535:
    labeled_neurites_size_filters2 = labeled_neurites_size_filters2.astype(np.uint16)
    

neurites_psyn_intensity = measure.regionprops_table(label_image=labeled_neurites_size_filters2, intensity_image=out_ch0_3d,properties=props)
neurites_psyn_intensity = np.asarray(pd.DataFrame.from_dict(neurites_psyn_intensity, orient='columns'))

neurites_map2_intensity = measure.regionprops_table(label_image=labeled_neurites_size_filters2, intensity_image=out_ch2_3d,properties=props)
neurites_map2_intensity = np.asarray(pd.DataFrame.from_dict(neurites_map2_intensity, orient='columns'))

neurites_nfh_intensity = measure.regionprops_table(label_image=labeled_neurites_size_filters2, intensity_image=out_ch1_3d,properties=props)
neurites_nfh_intensity = np.asarray(pd.DataFrame.from_dict(neurites_nfh_intensity, orient='columns'))
del labeled_neurites2
#%

map2_intensity_test = out_ch2_3d[map2_mask_image > 0]
nfh_intensity_test = out_ch1_3d[nfh_mask_image > 0]


map2_neurite_normalization_intensity = np.quantile(map2_intensity_test,q=mask_intensity_cutoff)
nfh_neurite_normalization_intensity = np.quantile(nfh_intensity_test,q=mask_intensity_cutoff)



map2_adjusted_intensity_image = np.where(out_ch2_3d > map2_neurite_normalization_intensity , out_ch2_3d ,0)

nfh_adjusted_intensity_image = np.where(out_ch1_3d > nfh_neurite_normalization_intensity , out_ch1_3d ,0)



map2_mask_image_raw = (map2_mask_image * np.where(out_ch2_3d > map2_neurite_normalization_intensity ,1 ,0)).astype(np.uint8)
nfh_mask_image_raw = (nfh_mask_image *  np.where(out_ch1_3d > nfh_neurite_normalization_intensity , 1 ,0)).astype(np.uint8)

make_negative_mask = True
if mask_type == 'negative':
    map2_mask_image = np.where(nfh_mask_image_raw == 0 ,map2_mask_image_raw ,0).astype(np.uint8)
    nfh_mask_image = np.where(map2_mask_image_raw == 0 ,nfh_mask_image_raw ,0).astype(np.uint8)
if mask_type == 'positive':
    map2_mask_image = map2_mask_image_raw.copy()
    nfh_mask_image = nfh_mask_image_raw.copy()




properties2 = measure.regionprops(label_image=labeled_neurites_size_filters2, intensity_image=map2_mask_image)# 
map2_coverage = []
all_tile_ids_unique = 0
for row in properties2:
    current_coords = []
    current_coverage = []
    current_intensity = []
    current_intensity_median = []
    current_intensity_mean = []
    current_neurite_tile_id = []
    current_coords = row.coords
    current_neurite_tile_id = np.unique(tile_id_map[current_coords[:,0],current_coords[:,1],current_coords[:,2]])
    if len(current_neurite_tile_id) != 1:
        print('neurite crosses border. Whats wrong?')
        print(row.label)
        all_tile_ids_unique = all_tile_ids_unique + 1
    current_coverage = map2_mask_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]] ##3/30/2026 drop _test
    ## chose either raw or normalizaed intensity image manually
    #current_intensity = out_ch2_3d[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    current_intensity = map2_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    current_intensity = current_intensity[current_coverage == 1]
    current_intensity_median = np.median(current_intensity)
    current_intensity_mean = np.mean(current_intensity)
    current_coverage = np.sum(current_coverage)
    current_intensity = np.sum(current_intensity)
    map2_coverage.append(np.hstack((row.label,current_coverage,current_intensity,current_intensity_median,current_intensity_mean,current_neurite_tile_id)))
    
map2_coverage = np.vstack(map2_coverage)
if all_tile_ids_unique == 0:
    print('all neurite tile ids were unique.')
                   
properties2 = measure.regionprops(label_image=labeled_neurites_size_filters2, intensity_image=nfh_mask_image)# 
nfh_coverage = []
all_tile_ids_unique = 0
for row in properties2:
    current_coords = []
    current_coverage = []
    current_intensity = []
    current_intensity_median = []
    current_intensity_mean = []
    current_neurite_tile_id = []
    current_coords = row.coords
    current_neurite_tile_id = np.unique(tile_id_map[current_coords[:,0],current_coords[:,1],current_coords[:,2]])
    if len(current_neurite_tile_id) != 1:
        print('neurite crosses border. Whats wrong?')
        print(row.label)
        all_tile_ids_unique = all_tile_ids_unique + 1
    current_coverage = nfh_mask_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]##3/30/2026 drop _test
    ## chose either raw or normalizaed intensity image manually
    #current_intensity = out_ch1_3d[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    current_intensity = nfh_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    current_intensity = current_intensity[current_coverage == 1]
    current_intensity_median = np.median(current_intensity)
    current_intensity_mean = np.mean(current_intensity)
    current_coverage = np.sum(current_coverage)
    current_intensity = np.sum(current_intensity)
    nfh_coverage.append(np.hstack((row.label,current_coverage,current_intensity,current_intensity_median,current_intensity_mean,current_neurite_tile_id)))
    
nfh_coverage = np.vstack(nfh_coverage)    
if all_tile_ids_unique == 0:
    print('all neurite tile ids were unique.')



test_alignment = np.sum(np.abs(neurites_psyn_intensity[:,0] - neurites_map2_intensity[:,0]))
test_alignment1 = np.sum(np.abs(neurites_psyn_intensity[:,0] - neurites_nfh_intensity[:,0]))
test_alignment2 = np.sum(np.abs(neurites_psyn_intensity[:,0] - map2_coverage[:,0]))
test_alignment3 = np.sum(np.abs(neurites_psyn_intensity[:,0] - nfh_coverage[:,0]))
#%
all_data=[]
if np.sum(test_alignment + test_alignment1 + test_alignment2 +test_alignment3) == 0:  
    all_data = np.hstack((neurites_psyn_intensity,
                          neurites_map2_intensity[:,2].reshape(np.shape(neurites_psyn_intensity)[0],1),
                          neurites_nfh_intensity[:,2].reshape(np.shape(neurites_psyn_intensity)[0],1),
                          map2_coverage[:,1:6].reshape(np.shape(neurites_psyn_intensity)[0],5),
                          nfh_coverage[:,1:6].reshape(np.shape(neurites_psyn_intensity)[0],5)))   



all_data_nan_to_zero = np.nan_to_num(all_data)                

key = np.hstack(((all_data_nan_to_zero[:,5] / all_data_nan_to_zero[:,10]).reshape(np.shape(all_data)[0],1),
                  (all_data_nan_to_zero[:,10] / all_data_nan_to_zero[:,5]).reshape(np.shape(all_data)[0],1)))

nfh_positive = ((key[:,0] < 0.5).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_positive = ((key[:,1] < 0.5).reshape(np.shape(all_data)[0],1)).astype(np.uint8)

all_data_nan_to_zero = np.hstack((all_data_nan_to_zero,key, map2_positive,nfh_positive))

### um^3 ovelap cutoff based on 200x35x35nm voxel rounded up to next whole voxel
map2_0 = ((all_data_nan_to_zero[:,5] > 0).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_5 = ((all_data_nan_to_zero[:,5] > 21).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_10 = ((all_data_nan_to_zero[:,5] > 41).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_15 = ((all_data_nan_to_zero[:,5] > 62).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_25 = ((all_data_nan_to_zero[:,5] > 103).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_50 = ((all_data_nan_to_zero[:,5] > 205).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
map2_100 = ((all_data_nan_to_zero[:,5] > 409).reshape(np.shape(all_data)[0],1)).astype(np.uint8)

nfh_0 = ((all_data_nan_to_zero[:,10] > 0).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_5 = ((all_data_nan_to_zero[:,10] > 21).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_10 = ((all_data_nan_to_zero[:,10] > 41).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_15 = ((all_data_nan_to_zero[:,10] > 62).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_25 = ((all_data_nan_to_zero[:,10] > 103).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_50 = ((all_data_nan_to_zero[:,10] > 205).reshape(np.shape(all_data)[0],1)).astype(np.uint8)
nfh_100 = ((all_data_nan_to_zero[:,10] > 409).reshape(np.shape(all_data)[0],1)).astype(np.uint8)


map2_pure_neurite = (all_data_nan_to_zero[:,15] == np.inf).reshape(np.shape(all_data)[0],1).astype(np.uint8)
nfh_pure_neurite = (all_data_nan_to_zero[:,16] == np.inf).reshape(np.shape(all_data)[0],1).astype(np.uint8)

all_data_nan_to_zero = np.hstack((all_data_nan_to_zero, map2_pure_neurite,nfh_pure_neurite,
                                  map2_0,map2_5,map2_10,map2_15,map2_25,map2_50,map2_100,
                                  nfh_0,nfh_5,nfh_10,nfh_15,nfh_25,nfh_50,nfh_100))


data_labels = np.asarray(['label','neurite_area','neurite_psn','neurite_map2','neurite_nfh',
'map2_voxel_overlap','map2_sum','map2_median','map2_mean','map2_tile_id',
'nfh_voxel_overlap','nfh_sum','nfh_median','nfh_mean','nfh_tile_id',
'map2/nfh','nfh/map2','map2_by_percent','nfh_by_percent','map2_pure','nfh_pure',
'map2_5um_overlap','map2_10um_overlap','map2_15um_overlap',
'nfh_5um_overlap','nfh_10um_overlap','nfh_15um_overlap']).reshape(1,27)

#%

out_ch0_3d = (out_ch0_3d / np.max(out_ch0_3d) * 255).astype(np.uint8)
out_ch1_3d = (out_ch1_3d / np.max(out_ch1_3d) * 255).astype(np.uint8)
out_ch2_3d = (out_ch2_3d / np.max(out_ch2_3d) * 255).astype(np.uint8)
out_neurite_mask_3d = (out_neurite_mask_3d / np.max(out_neurite_mask_3d) * 255).astype(np.uint8)
#%
unique_tile_ids = np.unique(all_data_nan_to_zero[:,9])
all_tile_map2_data = []
all_tile_nfh_data = []
neurite_psyn_intensity_map2 = []
neurite_psyn_intensity_nfh = []
process_i = [6,5,4,3,2,1,0]
map2_range_all = [21,22,23,24,25,26,27]
nfh_range_all = [28,29,30,31,32,33,34]
overlap_size_cutoff = [0,5,10,15,25,50,100]

## ADS 3/30/2026 adding steps to generate images of map2 and nfh masks
neurites_map2_positive = []
neurite_nfh_positive = [] 

by_tile_sum_map2 =[]
by_tile_mean_map2 = []
by_tile_median_map2 = []
by_tile_counts_map2 = []
by_tile_sum_nfh =[]
by_tile_mean_nfh = []
by_tile_median_nfh = []
by_tile_counts_nfh = []
mask_coverage_by_tile_all = []

for current_unique_tile_id in unique_tile_ids:
    
    mask_coverage_map2 = np.sum(map2_mask_image[tile_id_map == current_unique_tile_id])
    mask_coverage_nfh = np.sum(nfh_mask_image[tile_id_map == current_unique_tile_id])
    mask_coverage_by_tile_all.append(np.hstack((mask_coverage_map2,mask_coverage_nfh)).reshape(1,2))
        

    
    
    
    current_tile_subset = all_data_nan_to_zero[all_data_nan_to_zero[:,9] == current_unique_tile_id,:]
    if selection_type == 'pure':
        
        ## 17-20 selects based on amount of overlap for subtype selection
        current_tile_subset_pure_map2 = current_tile_subset[current_tile_subset[:,19] == 1 , :] ## 19= pure  map2, 17==map2 by percentlap
        current_tile_subset_pure_nfh = current_tile_subset[current_tile_subset[:,20] == 1 , :] ## 20 - pure nfh, 18= nfh by percent overlap
    if selection_type == 'percent':
        
        ## 17-20 selects based on amount of overlap for subtype selection
        current_tile_subset_pure_map2 = current_tile_subset[current_tile_subset[:,17] == 1 , :] ## 19= pure  map2, 17==map2 by percentlap
        current_tile_subset_pure_nfh = current_tile_subset[current_tile_subset[:,18] == 1 , :] ## 20 - pure nfh, 18= nfh by percent overlap
    if selection_type == 'any':
        
        ## 5 and 10 selects based on simple single voxel overlap detection
        current_tile_subset_pure_map2 = current_tile_subset[current_tile_subset[:,5] > 0 , :] ## 5 is map2 overlap
        current_tile_subset_pure_nfh = current_tile_subset[current_tile_subset[:,10] > 0  , :] ## 10 is nfh overlap
        
    
    by_tile_sum_map2.append(np.sum(current_tile_subset_pure_map2,axis=0).reshape(1,35))
    by_tile_mean_map2.append(np.mean(current_tile_subset_pure_map2,axis=0).reshape(1,35))
    by_tile_median_map2.append(np.median(current_tile_subset_pure_map2,axis=0).reshape(1,35))
    by_tile_sum_nfh.append(np.sum(current_tile_subset_pure_nfh,axis=0).reshape(1,35))
    by_tile_mean_nfh.append(np.mean(current_tile_subset_pure_nfh,axis=0).reshape(1,35))
    by_tile_median_nfh.append(np.median(current_tile_subset_pure_nfh,axis=0).reshape(1,35))
    by_tile_counts_map2.append(np.shape(current_tile_subset_pure_map2)[0])
    by_tile_counts_nfh.append(np.shape(current_tile_subset_pure_nfh)[0])

   
    
    
    ## ADS 3/30/2026 adding steps to generate images of map2 and nfh masks
    neurites_map2_positive.append(current_tile_subset_pure_map2[:,0])
    neurite_nfh_positive.append(current_tile_subset_pure_nfh[:,0])
    
    
      
    current_tile_subset_pure_map2_psyn = current_tile_subset_pure_map2[:,0:3]
    current_tile_subset_pure_nfh_psyn = current_tile_subset_pure_nfh[:,0:3]
    
    current_tile_subset_pure_map2_psyn = np.hstack((current_tile_subset_pure_map2_psyn.reshape(len(current_tile_subset_pure_map2_psyn),3),
                                                    np.repeat(current_unique_tile_id,len(current_tile_subset_pure_map2_psyn)).reshape(len(current_tile_subset_pure_map2_psyn),1)))
    current_tile_subset_pure_nfh_psyn = np.hstack((current_tile_subset_pure_nfh_psyn.reshape(len(current_tile_subset_pure_nfh_psyn),3),
                                                    np.repeat(current_unique_tile_id,len(current_tile_subset_pure_nfh_psyn)).reshape(len(current_tile_subset_pure_nfh_psyn),1)))
   
    neurite_psyn_intensity_map2.append(current_tile_subset_pure_map2_psyn)    
    neurite_psyn_intensity_nfh.append(current_tile_subset_pure_nfh_psyn)    
    subtract_counts = 0
    for map2_range_i in process_i:
        #print(map2_range_i)
        map2_range = map2_range_all[map2_range_i]
        current_cutoff_size = overlap_size_cutoff[map2_range_i]
        current_tile_subset_pure_map2_size_cutoff = current_tile_subset_pure_map2[current_tile_subset_pure_map2[:,map2_range] == 1]
        number_map2_pure_current_size = np.shape(current_tile_subset_pure_map2_size_cutoff)[0]
        mean_intensity_map2_pure_current_size = np.mean(current_tile_subset_pure_map2_size_cutoff[:,8])
        median_intensity_map2_pure_current_size = np.mean(current_tile_subset_pure_map2_size_cutoff[:,7])
        total_voxel_coverage_map2 = np.sum(current_tile_subset_pure_map2_size_cutoff[:,5])
        all_tile_map2_data.append(np.hstack((current_unique_tile_id,current_cutoff_size,number_map2_pure_current_size,
                                             (number_map2_pure_current_size - subtract_counts),
                                                 mean_intensity_map2_pure_current_size,
                                                 median_intensity_map2_pure_current_size,
                                                 total_voxel_coverage_map2)).reshape(1,7))
        subtract_counts = number_map2_pure_current_size
        #print(map2_range)
    
    subtract_counts = 0
    for nfh_range_i in process_i:
        nfh_range = nfh_range_all[nfh_range_i]
        current_cutoff_size = overlap_size_cutoff[nfh_range_i]
        current_tile_subset_pure_nfh_size_cutoff = current_tile_subset_pure_nfh[current_tile_subset_pure_nfh[:,nfh_range] == 1]
        number_nfh_pure_current_size = np.shape(current_tile_subset_pure_nfh_size_cutoff)[0]
        mean_intensity_nfh_pure_current_size = np.mean(current_tile_subset_pure_nfh_size_cutoff[:,13])
        median_intensity_nfh_pure_current_size = np.mean(current_tile_subset_pure_nfh_size_cutoff[:,12])
        total_voxel_coverage_nfh = np.sum(current_tile_subset_pure_nfh_size_cutoff[:,10])
        all_tile_nfh_data.append(np.hstack((current_unique_tile_id,current_cutoff_size,number_nfh_pure_current_size,
                                            (number_nfh_pure_current_size - subtract_counts),
                                                 mean_intensity_nfh_pure_current_size,
                                                 median_intensity_nfh_pure_current_size,
                                                 total_voxel_coverage_nfh)).reshape(1,7))
        subtract_counts = number_nfh_pure_current_size
        #print(nfh_range)
        
mask_coverage_by_tile_all = np.vstack(mask_coverage_by_tile_all)

by_tile_sum_map2 = np.vstack(by_tile_sum_map2)
by_tile_mean_map2 = np.vstack(by_tile_mean_map2)
by_tile_median_map2 = np.vstack(by_tile_median_map2)
by_tile_sum_nfh = np.vstack(by_tile_sum_nfh)
by_tile_mean_nfh = np.vstack(by_tile_mean_nfh)
by_tile_median_nfh = np.vstack(by_tile_median_nfh)

by_tile_counts_map2 = np.vstack(by_tile_counts_map2).reshape(len(by_tile_counts_map2 ),1)
by_tile_counts_nfh = np.vstack(by_tile_counts_nfh).reshape(len(by_tile_counts_nfh ),1)


#%
## ADS 3/30/2026 adding steps to generate images of map2 and nfh masks
neurites_map2_positive = np.hstack(neurites_map2_positive)
neurite_nfh_positive = np.hstack(neurite_nfh_positive)

percent_neurites_counts_by_tile_and_type = np.hstack(( (by_tile_counts_map2[:,0] / (by_tile_counts_map2[:,0] + by_tile_counts_nfh[:,0])  * 100).reshape(len(by_tile_sum_map2[:,5]),1),
                                               (by_tile_counts_nfh[:,0] / (by_tile_counts_map2[:,0] + by_tile_counts_nfh[:,0])  * 100).reshape(len(by_tile_sum_map2[:,5]),1)))

total_neurites_counts_by_tile = (by_tile_counts_map2[:,0] + by_tile_counts_nfh[:,0])

total_neurites_by_tile_voxels = (by_tile_sum_map2[:,5] + by_tile_sum_nfh[:,10]).reshape(len(by_tile_sum_map2[:,5]),1)

percent_neurites_voxels_by_tile_and_type = np.hstack(( (by_tile_sum_map2[:,5] / total_neurites_by_tile_voxels[:,0] * 100).reshape(len(by_tile_sum_map2[:,5]),1),
                                               (by_tile_sum_nfh[:,10] / total_neurites_by_tile_voxels[:,0] * 100).reshape(len(by_tile_sum_map2[:,5]),1)))

unlabeled_neurites = (all_data_nan_to_zero[:,0])[np.isin(all_data_nan_to_zero[:,0],np.hstack((neurites_map2_positive,neurite_nfh_positive)),invert=True)]
double_labeled_neurites = neurites_map2_positive[np.isin(neurites_map2_positive,neurite_nfh_positive)]
double_labeled_neurites2 = neurite_nfh_positive[np.isin(neurite_nfh_positive,neurites_map2_positive)]

#%
properties_intensity = measure.regionprops(label_image=labeled_neurites_size_filters2, intensity_image=out_ch0_3d)# 
all_neurites_coverage = []
all_tile_ids_unique = 0
for row in properties_intensity:
    current_coords = []
    current_coverage = []
    current_intensity = []
    current_intensity_sum = []
    current_intensity_median = []
    current_intensity_mean = []
    current_neurite_tile_id = []
    current_coords = row.coords
    current_neurite_tile_id = np.unique(tile_id_map[current_coords[:,0],current_coords[:,1],current_coords[:,2]])
    if len(current_neurite_tile_id) != 1:
        print('neurite crosses border. Whats wrong?')
        print(row.label)
        all_tile_ids_unique = all_tile_ids_unique + 1
    type_test = 0
    if np.isin(row.label,unlabeled_neurites):
        type_test = type_test + 1
        current_coverage = [0,0]
        current_intensity = [0,0]
        
    if np.isin(row.label,neurites_map2_positive):
        type_test = type_test + 2
        current_coverage = map2_mask_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]] ##3/30/2026 drop _test
        current_intensity = map2_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
        
            
    if np.isin(row.label,neurite_nfh_positive):
        type_test = type_test + 4
        current_coverage = nfh_mask_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]] ##3/30/2026 drop _test
        current_intensity = nfh_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    
    current_intensity = current_intensity[current_coverage == 1]
    current_intensity_median = np.median(current_intensity)
    current_intensity_mean = np.mean(current_intensity)
    current_coverage = np.sum(current_coverage)
    current_intensity_sum = np.sum(current_intensity)
    
    total_map2_intensity = map2_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    total_map2_intensity_mean = np.mean(total_map2_intensity)
    total_map2_intensity_median = np.median(total_map2_intensity)
    total_map2_intensity_sum = np.sum(total_map2_intensity)
    
    total_nfh_intensity = nfh_adjusted_intensity_image[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
    total_nfh_intensity_mean = np.mean(total_nfh_intensity)
    total_nfh_intensity_median = np.median(total_nfh_intensity)
    total_nfh_intensity_sum = np.sum(total_nfh_intensity)
    
    current_coords_min = np.min(current_coords, axis=0)
    current_coords_max = np.max(current_coords, axis=0)
    current_coords_planes = current_coords_max - current_coords_min
    current_planes_min = np.min(current_coords_planes)
    major_axis = [np.nan]
    minor_axis = [np.nan]
    axis_ratio = [np.nan]
    
    if current_planes_min >= 1:
        major_axis = row.axis_major_length
        minor_axis = row.axis_minor_length
        axis_ratio = major_axis / minor_axis

    
    
    
    
    
    all_neurites_coverage.append(np.hstack((row.label,current_neurite_tile_id,type_test,row.intensity_mean,row.area,major_axis,minor_axis,
                                    axis_ratio,
                                    current_coverage,current_intensity_sum,current_intensity_median,current_intensity_mean,
                                    total_map2_intensity_mean,total_map2_intensity_median,total_map2_intensity_sum,
                                    total_nfh_intensity_mean,total_nfh_intensity_median,total_nfh_intensity_sum)).reshape(1,18))
    #print(row.area)
all_neurites_coverage = np.vstack(all_neurites_coverage)
if all_tile_ids_unique == 0:
    print('all neurite tile ids were unique.')


#%
labeled_neurites_size_filters2_map2 = np.isin(labeled_neurites_size_filters2,neurites_map2_positive)
labeled_neurites_size_filters2_nfh = np.isin(labeled_neurites_size_filters2,neurite_nfh_positive)
    
all_tile_map2_data = np.vstack(all_tile_map2_data)
all_tile_nfh_data = np.vstack(all_tile_nfh_data)
neurite_psyn_intensity_map2 = np.vstack(neurite_psyn_intensity_map2)    
neurite_psyn_intensity_nfh = np.vstack(neurite_psyn_intensity_nfh)    

#%
labeled_neurites_size_filters2_save = (np.where(labeled_neurites_size_filters2>0,255,0)).astype(np.uint8)
labeled_neurites_size_filters2_map2 = (labeled_neurites_size_filters2_map2 * 255).astype(np.uint8)
labeled_neurites_size_filters2_nfh = (labeled_neurites_size_filters2_nfh * 255).astype(np.uint8)
#%
output_image = np.zeros((np.shape(out_ch0_3d)[0],8,np.shape(out_ch0_3d)[1],np.shape(out_ch0_3d)[2]),dtype=np.uint8)#.astype(np.uint8)
#%
output_image[:,0,:,:] = out_ch0_3d
output_image[:,1,:,:] = labeled_neurites_size_filters2_save
output_image[:,2,:,:] = out_ch2_3d
output_image[:,3,:,:] = (map2_mask_image * 255 * somal_mask_image_large).astype(np.uint8)
output_image[:,4,:,:] = labeled_neurites_size_filters2_map2
output_image[:,5,:,:] = out_ch1_3d
output_image[:,6,:,:] = (nfh_mask_image * 255 * somal_mask_image_large).astype(np.uint8)
output_image[:,7,:,:] = labeled_neurites_size_filters2_nfh


max_projection = output_max_projections(output_image)


tifffile.imwrite(outfolder + file[89:-4] + '_'+ mask_type +'_masks_map2_'+ selection_type + '_' + str(psyn_size_max) + suffix + '.ome.tif',max_projection[2:5,:,:],metadata={'axes': 'CYX','Pixels': {
            'PhysicalSizeY':0.035,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeX':0.035,
            'PhysicalSizeXUnit': 'µm',
            }})
tifffile.imwrite(outfolder + file[89:-4] + '_'+ mask_type +'_masks_nfh_'+ selection_type + '_' + str(psyn_size_max) + suffix + '.ome.tif',max_projection[5:8,:,:],metadata={'axes': 'CYX','Pixels': {
            'PhysicalSizeY':0.035,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeX':0.035,
            'PhysicalSizeXUnit': 'µm',
            }})
tifffile.imwrite(outfolder + file[89:-4] + '_'+ mask_type +'_masks_merged_masks_'+ selection_type + '_' + str(psyn_size_max) + suffix + '.ome.tif',max_projection[[3,6],:,:],metadata={'axes': 'CYX','Pixels': {
            'PhysicalSizeY':0.035,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeX':0.035,
            'PhysicalSizeXUnit': 'µm',
            }})
tifffile.imwrite(outfolder + file[89:-4] + '_'+ mask_type +'_masks_psyn_'+ selection_type + '_' + str(psyn_size_max) + suffix + '.ome.tif',max_projection[0:2,:,:],metadata={'axes': 'CYX','Pixels': {
            'PhysicalSizeY':0.035,
            'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeX':0.035,
            'PhysicalSizeXUnit': 'µm',
            }})


#%%
##############################################################
## Perform data reorganization and binning by size or animal
##############################################################


all_organized_data_by_tile_and_size = []

organize_i_range = [0,1,2,3,4,5,6]
for current_unique_tile_id in unique_tile_ids:
    for organize_range_i in organize_i_range:
        current_size_organize = overlap_size_cutoff[organize_range_i]
        current_map2_organize = all_tile_map2_data[all_tile_map2_data[:,0] == current_unique_tile_id,:]
        current_map2_organize = current_map2_organize[current_map2_organize[:,1] == current_size_organize,:]

        current_nfh_organize = all_tile_nfh_data[all_tile_nfh_data[:,0] == current_unique_tile_id,:]
        current_nfh_organize = current_nfh_organize[current_nfh_organize[:,1] == current_size_organize,:]
        
        total_labeled_neurites = np.sum(current_map2_organize[0,3] + current_nfh_organize[0,3])
        percent_map2 = current_map2_organize[0,3] / total_labeled_neurites *  100
        percent_nfh = current_nfh_organize[0,3] / total_labeled_neurites *  100
        
        total_voxel_coverage_map2_by_tile = current_map2_organize[0,6]
        total_voxel_coverage_nfh_by_tile = current_nfh_organize[0,6]
        
        current_organized_data = np.hstack((current_unique_tile_id,current_size_organize,total_labeled_neurites,
                                            current_map2_organize[0,3],current_nfh_organize[0,3],
                                            percent_map2, percent_nfh,
                                            total_voxel_coverage_map2_by_tile,total_voxel_coverage_nfh_by_tile)).reshape(1,9)
        all_organized_data_by_tile_and_size.append(current_organized_data)
all_organized_data_by_tile_and_size = np.vstack(all_organized_data_by_tile_and_size)

all_organized_data_by_tile = []
for current_unique_tile_id in unique_tile_ids:
    current_tile_data = all_organized_data_by_tile_and_size[all_organized_data_by_tile_and_size[:,0] ==current_unique_tile_id,:]
    current_tile_data = np.sum(current_tile_data,axis=0)
    all_organized_data_by_tile.append(np.hstack((current_unique_tile_id,current_tile_data[2:5],
                                                 (current_tile_data[3] / current_tile_data[2] * 100),
                                                 (current_tile_data[4] / current_tile_data[2] * 100),
                                                 current_tile_data[7],
                                                 current_tile_data[8])).reshape(1,8))
    
all_organized_data_by_tile = np.vstack(all_organized_data_by_tile)

### 3/31/2026 ADS warning this includes duplicate math due to size subselection overlap
organize_i_range = [0,1,2,3,4,5,6]
all_organized_data_by_animal = []
for organize_range_i in organize_i_range:
    current_size_organize = overlap_size_cutoff[organize_range_i]
    current_map2_organize = all_tile_map2_data[all_tile_map2_data[:,1] == current_size_organize,:]
    #current_map2_organize = current_map2_organize[current_map2_organize[:,1] == current_size_organize,:]

    current_nfh_organize = all_tile_nfh_data[all_tile_nfh_data[:,1] == current_size_organize,:]
    #current_nfh_organize = current_nfh_organize[current_nfh_organize[:,1] == current_size_organize,:]
    
    total_labeled_neurites = np.sum(current_map2_organize[:,3] + current_nfh_organize[:,3])
    percent_map2 = np.sum(current_map2_organize[:,3]) / total_labeled_neurites *  100
    percent_nfh = np.sum(current_nfh_organize[:,3]) / total_labeled_neurites *  100
    
    total_voxel_coverage_map2_by_animal = np.sum(current_map2_organize[:,6])
    total_voxel_coverage_nfh_by_animal = np.sum(current_nfh_organize[:,6])
    
    
    current_organized_data = np.hstack((current_size_organize,total_labeled_neurites,
                                        np.sum(current_map2_organize[:,3]),np.sum(current_nfh_organize[:,3]),
                                        percent_map2,percent_nfh,
                                        total_voxel_coverage_map2_by_animal,
                                        total_voxel_coverage_nfh_by_animal)).reshape(1,8)
    all_organized_data_by_animal.append(current_organized_data)
all_organized_data_by_animal = np.vstack(all_organized_data_by_animal)
#%%
##################################################
## Optional code to perform randomization analysis
## User just needs to run this block
##################################################

map2_mask_image_rotations = map2_mask_image.copy()
nfh_mask_image_rotations = nfh_mask_image.copy()
map2_adjusted_intensity_image_rotations = np.where(out_ch2_3d > map2_neurite_normalization_intensity , out_ch2_3d ,0)
nfh_adjusted_intensity_image_rotations = np.where(out_ch1_3d > nfh_neurite_normalization_intensity , out_ch1_3d ,0)


all_rotation_data = []
properties2 = measure.regionprops(label_image=labeled_neurites_size_filters2, intensity_image=map2_mask_image_rotations)# 
for rotation_z_i in range(0,2):
    for rotation_xy_i in range(0,5):
    
        if rotation_xy_i > 0:
            map2_mask_image_rotations = np.rot90(map2_mask_image_rotations,  k=1,axes=(1, 2))
            map2_adjusted_intensity_image_rotations = np.rot90(map2_adjusted_intensity_image_rotations,  k=1,axes=(1, 2))
            nfh_mask_image_rotations = np.rot90(nfh_mask_image_rotations,  k=1,axes=(1, 2))
            nfh_adjusted_intensity_image_rotations = np.rot90(nfh_adjusted_intensity_image_rotations,  k=1,axes=(1, 2))
        if rotation_z_i > 0:
            map2_mask_image_rotations = np.flipud(map2_mask_image_rotations)
            map2_adjusted_intensity_image_rotations = np.flipud(map2_adjusted_intensity_image_rotations)
            nfh_mask_image_rotations = np.flipud(nfh_mask_image_rotations)
            nfh_adjusted_intensity_image_rotations = np.flipud(nfh_adjusted_intensity_image_rotations)
             
        
        map2_coverage = []
        all_tile_ids_unique = 0
        for row in properties2:
            current_coords = []
            current_coverage = []
            current_intensity = []
            current_intensity_median = []
            current_intensity_mean = []
            current_neurite_tile_id = []
            current_coords = row.coords
            current_neurite_tile_id = np.unique(tile_id_map[current_coords[:,0],current_coords[:,1],current_coords[:,2]])
            if len(current_neurite_tile_id) != 1:
                print('neurite crosses border. Whats wrong?')
                print(row.label)
                all_tile_ids_unique = all_tile_ids_unique + 1
            current_coverage = map2_mask_image_rotations[current_coords[:,0],current_coords[:,1],current_coords[:,2]] ##3/30/2026 drop _test
            ## chose either raw or normalizaed intensity image manually
            #current_intensity = out_ch2_3d[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
            current_intensity = map2_adjusted_intensity_image_rotations[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
            current_intensity = current_intensity[current_coverage == 1]
            current_intensity_median = np.median(current_intensity)
            current_intensity_mean = np.mean(current_intensity)
            current_coverage = np.sum(current_coverage)
            current_intensity = np.sum(current_intensity)
            map2_coverage.append(np.hstack((row.label,current_coverage,current_intensity,current_intensity_median,current_intensity_mean,current_neurite_tile_id)))
            #print(row.area)
        map2_coverage = np.vstack(map2_coverage)
        
        
        if all_tile_ids_unique == 0:
            print('all neurite tile ids were unique.')
                           
        properties2 = measure.regionprops(label_image=labeled_neurites_size_filters2, intensity_image=nfh_mask_image_rotations)# 
        nfh_coverage = []
        all_tile_ids_unique = 0
        for row in properties2:
            current_coords = []
            current_coverage = []
            current_intensity = []
            current_intensity_median = []
            current_intensity_mean = []
            current_neurite_tile_id = []
            current_coords = row.coords
            current_neurite_tile_id = np.unique(tile_id_map[current_coords[:,0],current_coords[:,1],current_coords[:,2]])
            if len(current_neurite_tile_id) != 1:
                print('neurite crosses border. Whats wrong?')
                print(row.label)
                all_tile_ids_unique = all_tile_ids_unique + 1
            current_coverage = nfh_mask_image_rotations[current_coords[:,0],current_coords[:,1],current_coords[:,2]]##3/30/2026 drop _test
            ## chose either raw or normalizaed intensity image manually
            #current_intensity = out_ch1_3d[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
            current_intensity = nfh_adjusted_intensity_image_rotations[current_coords[:,0],current_coords[:,1],current_coords[:,2]]
            current_intensity = current_intensity[current_coverage == 1]
            current_intensity_median = np.median(current_intensity)
            current_intensity_mean = np.mean(current_intensity)
            current_coverage = np.sum(current_coverage)
            current_intensity = np.sum(current_intensity)
            nfh_coverage.append(np.hstack((row.label,current_coverage,current_intensity,current_intensity_median,current_intensity_mean,current_neurite_tile_id)))
            #print(row.area)
        nfh_coverage = np.vstack(nfh_coverage)    
        if all_tile_ids_unique == 0:
            print('all neurite tile ids were unique.')
        
        
        
        test_alignment = np.sum(np.abs(neurites_psyn_intensity[:,0] - neurites_map2_intensity[:,0]))
        test_alignment1 = np.sum(np.abs(neurites_psyn_intensity[:,0] - neurites_nfh_intensity[:,0]))
        test_alignment2 = np.sum(np.abs(neurites_psyn_intensity[:,0] - map2_coverage[:,0]))
        test_alignment3 = np.sum(np.abs(neurites_psyn_intensity[:,0] - nfh_coverage[:,0]))
        #%
        all_data=[]
        if np.sum(test_alignment + test_alignment1 + test_alignment2 +test_alignment3) == 0:  
            all_data = np.hstack((neurites_psyn_intensity,
                                  neurites_map2_intensity[:,2].reshape(np.shape(neurites_psyn_intensity)[0],1),
                                  neurites_nfh_intensity[:,2].reshape(np.shape(neurites_psyn_intensity)[0],1),
                                  map2_coverage[:,1:6].reshape(np.shape(neurites_psyn_intensity)[0],5),
                                  nfh_coverage[:,1:6].reshape(np.shape(neurites_psyn_intensity)[0],5)))   
        
        
        
        all_data_nan_to_zero = np.nan_to_num(all_data)  
        unique_tiles_randomization = np.unique(all_data_nan_to_zero[:,9])              
        for current_unique_tile_randomization in unique_tiles_randomization:
            current_unique_tile_randomization_data = all_data_nan_to_zero[all_data_nan_to_zero[:,9] == current_unique_tile_randomization]
            key = np.hstack(((current_unique_tile_randomization_data[:,5] / current_unique_tile_randomization_data[:,10]).reshape(np.shape(current_unique_tile_randomization_data)[0],1),
                              (current_unique_tile_randomization_data[:,10] / current_unique_tile_randomization_data[:,5]).reshape(np.shape(current_unique_tile_randomization_data)[0],1)))
            
            total_neurites = np.shape(key)[0]
            nfh_positive_sum_voxels = np.sum(current_unique_tile_randomization_data[key[:,0] < 0.5,10])
            map2_positive_sum_voxels = np.sum(current_unique_tile_randomization_data[key[:,1] < 0.5,5]) 
            nfh_pure_neurite_sum_voxels = np.sum(current_unique_tile_randomization_data[key[:,1] == np.inf,10])
            map2_pure_neurite_sum_voxels = np.sum(current_unique_tile_randomization_data[key[:,0] == np.inf,5])
            
            nfh_any_neurite_sum_voxels = np.sum(current_unique_tile_randomization_data[current_unique_tile_randomization_data[:,10] >0,10])
            map2_any_neurite_sum_voxels = np.sum(current_unique_tile_randomization_data[current_unique_tile_randomization_data[:,5] >0,5])
            
            
            nfh_positive__sum_counts = len(key[key[:,0] < 0.5,0])
            map2_positive__sum_counts = len(key[key[:,1] < 0.5,0])
            
            nfh_pure_neurite__sum_counts = len(key[key[:,1] == np.inf,0])
            map2_pure_neurite__sum_counts = len(key[key[:,0] == np.inf,0])
            
            nfh_any_neurite__sum_counts = len(current_unique_tile_randomization_data[current_unique_tile_randomization_data[:,10] >0,0])
            map2_any_neurite__sum_counts = len(current_unique_tile_randomization_data[current_unique_tile_randomization_data[:,5] >0,0])
            
       

        
        
            current_rotation_data = np.hstack((rotation_z_i,rotation_xy_i,total_neurites,map2_positive_sum_voxels,nfh_positive_sum_voxels,
                                               map2_pure_neurite_sum_voxels,nfh_pure_neurite_sum_voxels,
                                               map2_any_neurite_sum_voxels,nfh_any_neurite_sum_voxels,
                                               map2_positive__sum_counts,nfh_positive__sum_counts,
                                               map2_pure_neurite__sum_counts,nfh_pure_neurite__sum_counts,
                                               map2_any_neurite__sum_counts,nfh_any_neurite__sum_counts,
                                               current_unique_tile_randomization)).reshape(1,16)
            all_rotation_data.append(current_rotation_data)
all_rotation_data = np.vstack(all_rotation_data)
#%
total_percent_positive_voxels = np.sum(all_rotation_data[:,3:5],axis=1).reshape(160,1)
total_pure_positive_voxels = np.sum(all_rotation_data[:,5:7],axis=1).reshape(160,1)
total_any_positive_voxels = np.sum(all_rotation_data[:,7:9],axis=1).reshape(160,1)

total_percent_positive_voxels = np.hstack((total_percent_positive_voxels,total_percent_positive_voxels))
total_pure_positive_voxels = np.hstack((total_pure_positive_voxels,total_pure_positive_voxels))
total_any_positive_voxels = np.hstack((total_any_positive_voxels ,total_any_positive_voxels))

total_percent_positive_voxels = ((all_rotation_data[:,3:5]) / total_percent_positive_voxels)*100
total_pure_positive_voxels = ((all_rotation_data[:,5:7]) / total_pure_positive_voxels)*100
total_any_positive_voxels = ((all_rotation_data[:,7:9]) / total_any_positive_voxels)*100

