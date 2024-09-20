# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:50:17 2022

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
    color, feature, filters, measure, morphology, segmentation, util)
from aicsimageio import AICSImage    
    
from PIL import Image, ImageEnhance
from skimage.transform import rescale, resize 
from matplotlib.patches import Rectangle   
#%%
#########################################
## User defined required input parameters
#########################################

####################################################
## Define input and output folders
####################################################
## Location and filename of master key for bulk processing tilescans
input_file_key = 'C:/file_key_roi_processing_heatmaps.csv'
## Output folder. Folder must be manually created and exist before processing.
output_folder_name = 'C:/heatmaps/'
## Location and filename of label.text file for processing Atlas Registration
labelfile = 'C:/labels.txt'

################################################
## Spacing gaps in Airyscan Processed Tilescans.
###############################################
## Data collected on a Zeiss 980 using the SR-8Y Multiplex mode is 12
tile_zero_spacing = 12

#########################################################
## Define radial smoothing and tile scaling if needed
## Define third_channel background substraction if needed
#########################################################
## Define radius if smoothing heatmaps is required
radius = 0
## Define scaline factor to subdivide image tiles. If left as full XY tile size then resulting data will be based on entire tile.
scaling_factor = 1748
## If third channel baseline substraction needs to be applied after running initial SEQUIN_v15 code.
pSYN_bls = 0
## Specific rows in input_file_key to process if sub-selecting tilescans. Numbering corresponds to row # listed in the first column of input_file_key.
file_i_to_process =[5,11,15,21,25,27]

##############################################################
## Define location of key with manually defined area selection
## Define output folder location for overlays of cropped area
############################################################

## Location and filename for manual subregion selection coordinates
input_file_key_crops = 'C:/box_crop_coordsv3.csv'
## Output folder name for reference images to visually confirm manual subregion selection was appropriate.
## Folder must be manually created and exist before processing.
output_folder_name = 'C:/heatmaps/cropsv3_r0s1748/'
## File format for output images
image_format = 'png' # e.g .png, .svg, etc.
## File naming definition for output images
def filenaming(i):
    image_name = 'slide_' + input_file_key_data[i,1] + 'sec_' + input_file_key_data[i,2] + '_' + input_file_key_data[i,3] + '_' + input_file_key_data[i,10] + '_' + input_file_key_data[i,11] + 'crop_box.png'
    return image_name

#%%
###############################################################
## Loads and processes intial input data.
## Subcrops data and stores output and variable in python for extraction
## Just run this block of code.
###############################################################

input_file_key_data = np.asarray(dask.dataframe.read_csv(input_file_key, delimiter=',', header=None)) 
input_file_key_data_header = input_file_key_data[0,:]
input_file_key_data = input_file_key_data[1::,1::]


density_heatmaps = [None] * np.shape(input_file_key_data)[0]
third_channel_raw_heatmaps = [None] * np.shape(input_file_key_data)[0]
third_channel_flip_heatmaps = [None] * np.shape(input_file_key_data)[0]
reference_image_storage = [None] * np.shape(input_file_key_data)[0]
neurite_distance_heatmaps_counts_positive = [None] * np.shape(input_file_key_data)[0]
neurite_distance_heatmaps_average_intensity = [None] * np.shape(input_file_key_data)[0]
neurite_distance_detected = [None] * np.shape(input_file_key_data)[0]
original_tilescan_pixels = [None] * np.shape(input_file_key_data)[0]
section_restricted_mask_full_tiles = [None] * np.shape(input_file_key_data)[0]
raw_pSYN_intensity = [None] * np.shape(input_file_key_data)[0]
raw_pSYN_intensity_small = [None] * np.shape(input_file_key_data)[0]
cropped_roi_masks = [None] * np.shape(input_file_key_data)[0]
tile_ids = [None] * np.shape(input_file_key_data)[0]
tile_ids_v2 = [None] * np.shape(input_file_key_data)[0]


def GetCZImetadata(current_row_file,tilescan_TR_id):
    metadatadict_czi =(czi.CziFile(current_row_file)).metadata(raw=False)
    metadatadict_czi_table =(czi.CziFile(current_row_file)).metadata(raw=True)
    #Find current image tile name
    curent_image_tile_ID = re.split('-',current_row_file)
    curent_image_tile_ID = curent_image_tile_ID[-1]
    curent_image_tile_ID = curent_image_tile_ID[0:-4]
    number_of_TRs = len(metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              [0]
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion'])
    all_TR_names = []
    for current_TR in range(0,number_of_TRs):
        all_TR_names.append(metadatadict_czi['ImageDocument']
                  ['Metadata']
                  ['Experiment']
                  ['ExperimentBlocks']
                  ['AcquisitionBlock']
                  [0]
                  ['SubDimensionSetups']
                  ['RegionsSetup']
                  ['SampleHolder']
                  ['TileRegions']
                  ['TileRegion']
                  [current_TR]
                  ['Name'])
    all_TR_names = np.asarray(all_TR_names)
    tilescan_position = int(np.asarray(np.where(all_TR_names == tilescan_TR_id)))
    current_tile_columns = (metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              [0]
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion']
              [tilescan_position]
              ['Columns'])
              
    current_tile_rows = (metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              [0]
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion']
              [tilescan_position]
              ['Rows'])        

    current_tile_SizeX = (metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Information']
              ['Image']
              ['SizeX'])
              
    current_tile_SizeY = (metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Information']
              ['Image']
              ['SizeY'])
    


    return current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY,metadatadict_czi, metadatadict_czi_table






#%


def generate_heatmaps(i):
    #%
 #%
    #i = 5
    # load 3rd channel reference image from prior processing of tilescan projections
    third_channel_image = input_file_key_data[i,15]
    third_channel_image = AICSImage(third_channel_image)
    third_channel_image = third_channel_image.get_image_data()
    third_channel_image = third_channel_image[0,0,0,:,:]
    third_channel_image = np.where(third_channel_image>pSYN_bls,third_channel_image - pSYN_bls,0)
    
    # load data after filtering synaptic subset and indentifying roi from registered atlas
    precompiled_input = input_file_key_data[i,14]
    precompiled_input_data = np.asarray(dask.dataframe.read_csv(precompiled_input, delimiter=',', header=None)) 
    precompiled_input_data = precompiled_input_data[1::,1::]
    precompiled_input_data = precompiled_input_data[precompiled_input_data[:,12] < .5]
    precompiled_input_data = precompiled_input_data[precompiled_input_data[:,16] == 'synaptic']
    
    # raw czi image file
    tilescan_image = input_file_key_data[i,3]
    # manually indentified tileregion number corresponding to czi file
    tilescan_TR_id = input_file_key_data[i,4]#'TR1'
    ## Airyscan post processing spacing. 980 8Y is 12.
    tile_zero_spacing = 12
    
    # max intensity projection for manual segmenting of tile region for processing
    reference_image = input_file_key_data[i,13]
    reference_image = AICSImage(reference_image)
    reference_image = reference_image.get_image_data()
    reference_image = reference_image[0,0,0,:,:,:]
    
    # manually traced section masks to exclude tilescan regions that aren't needed
    section_restricted_mask = input_file_key_data[i,7]
    section_restricted_mask = AICSImage(section_restricted_mask) 
    section_restricted_mask = section_restricted_mask.get_image_dask_data("YX")  # returns out-of-memory 4D dask array
    section_restricted_mask = section_restricted_mask.compute()
    section_restricted_mask = np.where(section_restricted_mask==255,1,0).astype(np.uint16)
    
    # if manual baseline subtraction is needed
    perform_post_bls = False
    post_bls_cutoff = 150
    
    
    #roi_mask = 'F:/Backed up/ASAP/yale-tilescans/asap tilescan reference images/sec3_50pNonL1.png'
    #roi_mask = AICSImage(roi_mask)
    #roi_mask = roi_mask.get_image_data("YX")
    #roi_mask = np.where(roi_mask == 255,0,1)
    
    # Automatically build mosaic tile key
    czi_metadata = czi.CziFile(tilescan_image)
    directory = czi_metadata.filtered_subblock_directory
    mosaic_key = []
    for directory_entry in directory:
        mosaic_index = directory_entry.mosaic_index
        mosaic_start = directory_entry.start
        mosaic_key.append(np.hstack((mosaic_index,mosaic_start)))
    mosaic_key = np.vstack(mosaic_key)
    ## Select slice 0
    mosaic_key = mosaic_key[mosaic_key[:,4]==0]
    ## Select channel 0
    mosaic_key = mosaic_key[mosaic_key[:,3]==0]
    
    mosaic_key_tile_ids = (mosaic_key[:,5:7] / 1760).astype(np.uint16)
    tile_id_map_v2 = np.zeros((int(np.max(mosaic_key_tile_ids[:,0])+1),int(np.max(mosaic_key_tile_ids[:,1])+1)))
    tile_id_map_v2[mosaic_key_tile_ids[:,0],mosaic_key_tile_ids[:,1]] = mosaic_key[:,0]
    
    tile_ids_v2[i] = tile_id_map_v2
    #heatmap = out_ch0.copy()
    img = AICSImage(tilescan_image,reconstruct_mosaic=False) 
    
    current_row_file = tilescan_image
    scene = 0
    current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY, current_metadatadict_czi, current_metadatadict_czi_table = GetCZImetadata(current_row_file,tilescan_TR_id)#,scene)
    #%
    
    
    lazy_t0 = img.get_image_dask_data("ZYX", M=0,C=0)  # returns out-of-memory 4D dask array
    t0 = lazy_t0.compute()
    t0 = t0[int((np.shape(t0)[0])/2),:,:]#np.max(t0,axis=0)
     
    actual_Y_tile_size = np.shape(t0)[0]
    actual_X_tile_size = np.shape(t0)[1] 
    
    
    metadatadict_czi_True = czi_metadata.metadata(raw=True)
    metadatadict_czi = czi_metadata.metadata(raw=False)
    
    
    xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
    z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
    voxelsize=[z_dim,xy_dim,xy_dim]
    
    
    
    
    
    img = AICSImage(tilescan_image,reconstruct_mosaic=False) 
    lazy_t0 = img.get_image_dask_data("ZYX", M=0,C=0)  # returns out-of-memory 4D dask array
    t0 = lazy_t0.compute()
    t0 = t0[int((np.shape(t0)[0])/2),:,:]#np.max(t0,axis=0)
    original_XY_tile_size = int(np.shape(t0)[0])
    actual_Y_tile_size = int(np.shape(t0)[0])
    actual_X_tile_size = int(np.shape(t0)[1])
    
    #%
    ## column 154 must be tile id # from tilescan
    mosaic_tile_ids = np.unique(precompiled_input_data[:,0])
    tile_coord_coorections = []
    for mosaic_tile_i in mosaic_tile_ids:
        y_coord = int(np.max((((mosaic_key[mosaic_key[:,0]==mosaic_tile_i,5])- ((mosaic_key[mosaic_key[:,0]==mosaic_tile_i,5] / ( actual_X_tile_size + tile_zero_spacing)) * tile_zero_spacing)),0)))
        x_coord = int(np.max((((mosaic_key[mosaic_key[:,0]==mosaic_tile_i,6])- ((mosaic_key[mosaic_key[:,0]==mosaic_tile_i,6] / ( actual_X_tile_size + tile_zero_spacing)) * tile_zero_spacing)),0)))
        tile_coord_coorections.append(np.hstack((mosaic_tile_i,y_coord,x_coord)))
    tile_coord_coorections=np.vstack(tile_coord_coorections)
    x_correct = precompiled_input_data[:,0]
    y_correct = precompiled_input_data[:,0]
    for mosaic_tile_i in mosaic_tile_ids:
        x_correct = np.where(precompiled_input_data[:,0] == mosaic_tile_i,tile_coord_coorections[tile_coord_coorections[:,0]==mosaic_tile_i,2],x_correct)
        y_correct = np.where(precompiled_input_data[:,0] == mosaic_tile_i,tile_coord_coorections[tile_coord_coorections[:,0]==mosaic_tile_i,1],y_correct)
    ##column 7 is weighted psd x coord and 53 is weighted syn x coord
    ##columm 6 is weighted psd y coords and 52 is weighted syn y coord
    synaptic_centroid_x_coords = ((precompiled_input_data[:,1] + precompiled_input_data[:,3]) /2).astype(float)
    synaptic_centroid_y_coords = ((precompiled_input_data[:,2] + precompiled_input_data[:,4]) /2).astype(float)
    
    synaptic_centroid_x_coords = synaptic_centroid_x_coords + x_correct.astype(float)
    synaptic_centroid_y_coords = synaptic_centroid_y_coords + y_correct.astype(float)
    
    
    synaptic_centroid_x_coords = np.round(synaptic_centroid_x_coords).astype(np.uint16)
    synaptic_centroid_y_coords = np.round(synaptic_centroid_y_coords).astype(np.uint16)
        
    #precompiled_input_data_267 = precompiled_input_data[precompiled_input_data[:,154]==267]
    ## 104 is distance to neurite
    ## 105 is max search radius for neurite
    ## if 104 is greater than 105 then no neurite was detected and difference in distance will be 0.001
    ## if 104-105 is negative then a neurite was detected
    no_neurite_detected = precompiled_input_data[:,5] - precompiled_input_data[:,6]
    neurite_detected_rows = np.where(no_neurite_detected<0,1,0)
#%  
    
    ##106 is synaptic centroif 200nm synaptic field 3rd channel intensity
    ##130 is synaptic centroid 200nm field from flipped 3rd channel
    all_coords_WMbls = np.hstack((precompiled_input_data[:,10:13],
                                  precompiled_input_data[:,5].reshape(np.shape(precompiled_input_data)[0],1),
                                   precompiled_input_data[:,7].reshape(np.shape(precompiled_input_data)[0],1),
                                   synaptic_centroid_y_coords.reshape(np.shape(precompiled_input_data)[0],1),
                                   synaptic_centroid_x_coords.reshape(np.shape(precompiled_input_data)[0],1),
                                   precompiled_input_data[:,13].reshape(np.shape(precompiled_input_data)[0],1),
                                   precompiled_input_data[:,8].reshape(np.shape(precompiled_input_data)[0],1),
                                   precompiled_input_data[:,5].reshape(np.shape(precompiled_input_data)[0],1),
                                   neurite_detected_rows.reshape(np.shape(precompiled_input_data)[0],1),
                                   precompiled_input_data[:,0].reshape(np.shape(precompiled_input_data)[0],1)))
    
        
    
    #%
    #current_file_metadata = np.asarray(dask.dataframe.read_csv(folder + '/' +  filename, delimiter=',', header=None)) 
    #current_file_metadata = current_file_metadata[1::,1::]
    
    
     
    
    #%
    ## all_coords_WMbls 5:7 is synaptic centroid calculated above
    all_coords_WMbls2 = all_coords_WMbls.copy()
    all_coords_WMbls2[:,5:7] = all_coords_WMbls2[:,5:7] / scaling_factor
    out_heatmap = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))
    intensity_3rd_channel_heatmap = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))
    #tile_id_map = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))
    #%
    metadatadict_czi = czi_metadata.metadata(raw=False)
    xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
    z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
    voxelsize=[z_dim,xy_dim,xy_dim]
    #%
    xy_radius_scale = 1
    #print(nm_cutoff)    
    #for radius in range(19,20):
                                                              
    #radius = 8 ##Manual override
    #radius_check = synaptic_field_voxel_radius ##If manuall switch to radius on line above    #### !!! Manually radius selection below needs to be changes
    #nm_cutoff = synaptic_field_nm_radius#(nm_cutoff_all[radius - 1])/1000
    #print(radius)
    #print(nm_cutoff)
    
    #radius = [5,10,20]
    # Create integers for coordinate adjustement to set maximum possible spot size
    coords = np.linspace(-(radius * xy_radius_scale),(radius * xy_radius_scale),num=(((radius * xy_radius_scale)*2)+1),dtype=int)
    coords = coords.astype(np.int16)
    pairs = np.asarray(np.meshgrid(coords,coords)).T.reshape(-1,2)
    
    X= np.sqrt((pairs[:,0]**2)+(pairs[:,1]**2))
    
    pairs = pairs[X<=radius]
   
    
    
    image_dims = np.shape(out_heatmap)
    synaptic_3rd_channel_heatmap_data = []
    synaptic_3rd_channel_heatmap_data_flip = []
    neurite_distance_array = []
    neurite_distance_array_250nm = []
    
    ## if all_coords_WMbls2[:,10]==1 then neurite was detection and column 9 is the distance the neurite was from synaptic loci
    detected_neurite_ditances = all_coords_WMbls2[all_coords_WMbls2[:,10]==1,9]
    neurite_distance_detected[i] = detected_neurite_ditances
    
    for r in range(0,np.shape(all_coords_WMbls2)[0]):
        all_coords_WMbls2[r,5] = np.round(all_coords_WMbls2[r,5])
        all_coords_WMbls2[r,6] = np.round(all_coords_WMbls2[r,6])
        
    out_heatmap = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))
    neurite_positive_heatmap = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))
    
    
    #if neurite_test in range(0,2):
    ## get synaptic loci with a detected neurite
    neurite_postive_loci = all_coords_WMbls2[all_coords_WMbls2[:,10]==1]
    ## reset max distance of neurite. 
    neurite_postive_loci2 = np.where(neurite_postive_loci[:,9] > 0.4,0.4,neurite_postive_loci[:,9])
    neurite_postive_loci = np.hstack((neurite_postive_loci,neurite_postive_loci2.reshape(len(neurite_postive_loci2),1)))
    neurite_negative_loci = all_coords_WMbls2[all_coords_WMbls2[:,10]==0]
    neurite_negative_loci2 = neurite_negative_loci[neurite_negative_loci[:,9]>.4]
    neurite_negative_loci2 = np.where(neurite_negative_loci[:,9] >  0.4,0.4,neurite_negative_loci[:,9])
    neurite_negative_loci = np.hstack((neurite_negative_loci,neurite_negative_loci2.reshape(len(neurite_negative_loci2),1)))
    neurite_distance_array_250nm = np.vstack((neurite_postive_loci,neurite_negative_loci))
    neurite_distance_array_250nm = (neurite_distance_array_250nm[:,[5,6,11]]).astype(float)
    
    neurite_distance_array_250nm = neurite_distance_array_250nm[neurite_distance_array_250nm[:,0] >=0 ]
    neurite_distance_array_250nm = neurite_distance_array_250nm[neurite_distance_array_250nm[:,0] <=(image_dims[0]-1) ]
    neurite_distance_array_250nm = neurite_distance_array_250nm[neurite_distance_array_250nm[:,1] >=0 ]
    neurite_distance_array_250nm = neurite_distance_array_250nm[neurite_distance_array_250nm[:,1] <=(image_dims[1]-1) ]
    #neurite_distance_array_250nm = neurite_distance_array_250nm.astype(np.uint16)
    
    
    
    
    #cumulative_average = np.zeros((int(current_tile_SizeY/100),int(current_tile_SizeX/100)))
    for row in all_coords_WMbls2:
        #print(row)
        coords = row[5:7].reshape(1,2) + pairs
        
        #current_synaptic_field_neurite_org = current_synaptic_field_neurite.copy()
       
        coords = coords[coords[:,0] >=0 ]
        coords = coords[coords[:,0] <=(image_dims[0]-1) ]
        coords = coords[coords[:,1] >=0 ]
        coords = coords[coords[:,1] <=(image_dims[1]-1) ]
        coords = coords.astype(np.uint16)
        out_heatmap[coords[:,0],coords[:,1]] = out_heatmap[coords[:,0],coords[:,1]] + 1
        #tile_id_map[coords[:,0],coords[:,1]] = row[11]
        if row[10] == 1:
            neurite_positive_heatmap[coords[:,0],coords[:,1]] = neurite_positive_heatmap[coords[:,0],coords[:,1]] + 1
        #current_3rd_channel_spot = np.zeros((int(current_tile_SizeY/100),int(current_tile_SizeX/100)))
        #current_3rd_channel_heatmap = np.zeros((int(current_tile_SizeY/100),int(current_tile_SizeX/100)))
        
        current_neurite_distance = np.repeat(float(row[9]),np.shape(coords)[0]).reshape(np.shape(coords)[0],1)
        current_neurite_distance = np.hstack((coords,current_neurite_distance))
        neurite_distance_array.append(current_neurite_distance)
        
        
        
        ## 4 is raw field data and 8 is flipped field data
        ## should consider replacing 8 with the average fieldwise non-neurite intensity from updated SEQUIN code
        if (float(row[4]) != 0) and  (float(row[8]) != 0):
            intensity = float(row[4]) / float(row[8])
            current_3rd_channel_heatmap_data = np.repeat(intensity,np.shape(coords)[0]).reshape(np.shape(coords)[0],1)
            current_3rd_channel_heatmap_data = np.hstack((coords,current_3rd_channel_heatmap_data))
            synaptic_3rd_channel_heatmap_data_flip.append(current_3rd_channel_heatmap_data)
    
            intensity = float(row[4]) 
            current_3rd_channel_heatmap_data = np.repeat(intensity,np.shape(coords)[0]).reshape(np.shape(coords)[0],1)
            current_3rd_channel_heatmap_data = np.hstack((coords,current_3rd_channel_heatmap_data))
            synaptic_3rd_channel_heatmap_data.append(current_3rd_channel_heatmap_data)
        #current_3rd_channel_spot[coords[:,0],coords[:,1]] = 1
        #intensity_3rd_channel_heatmap = ((intensity_3rd_channel_heatmap * cumulative_average) + current_3rd_channel_heatmap) / (cumulative_average + 1)
        #cumulative_average = cumulative_average + 1
    #%
    synaptic_3rd_channel_heatmap_data_flip = np.vstack(synaptic_3rd_channel_heatmap_data_flip)
    synaptic_3rd_channel_heatmap_data = np.vstack(synaptic_3rd_channel_heatmap_data)
    neurite_distance_array = np.vstack(neurite_distance_array)
    
    ## find and remove nan values
    test_nan_raw = np.invert(np.isnan(synaptic_3rd_channel_heatmap_data))[:,2]
    test_nan_flip = np.invert(np.isnan(synaptic_3rd_channel_heatmap_data_flip))[:,2]
    test_nan_neurite_distances = np.invert(np.isnan(neurite_distance_array))[:,2]
   
    #neurite_distance_array_250nm[:,11] = (neurite_distance_array_250nm[:,11]).astype(float)
    test_250nm_max_neurites = np.invert(np.isnan(neurite_distance_array_250nm))[:,2]
    
    
    
    
    
    synaptic_3rd_channel_heatmap_data = synaptic_3rd_channel_heatmap_data[test_nan_raw,:]
    synaptic_3rd_channel_heatmap_data_flip = synaptic_3rd_channel_heatmap_data_flip[test_nan_flip,:]
    neurite_distance_array = neurite_distance_array[test_nan_neurite_distances,:]
    neurite_distance_array_250nm = neurite_distance_array_250nm[test_250nm_max_neurites]
    
    
    synaptic_3rd_channel_heatmap_data_df = pd.DataFrame(synaptic_3rd_channel_heatmap_data)
    synaptic_3rd_channel_heatmap_data_flip_df = pd.DataFrame(synaptic_3rd_channel_heatmap_data_flip)
    neurite_distance_array_df = pd.DataFrame(neurite_distance_array)
    neurite_distance_array_250nm_df = pd.DataFrame(neurite_distance_array_250nm)
    
    
    
    #%
    means = synaptic_3rd_channel_heatmap_data_df.groupby([0,1]).mean()
    means_flip = synaptic_3rd_channel_heatmap_data_flip_df.groupby([0,1]).mean()
    means_neurites = neurite_distance_array_df.groupby([0,1]).mean()
    means_neurites_250 = neurite_distance_array_250nm_df.groupby([0,1]).mean()
    #%
    index = means.index.values
    index = np.vstack(index)
    
    index_flip = means_flip.index.values
    index_flip = np.vstack(index_flip)
    
    index_neurites = means_neurites.index.values
    index_neurites = np.vstack(index_neurites)
    
    index_neurites_250 = means_neurites_250.index.values
    index_neurites_250 = np.vstack(index_neurites_250)
    
    
    #%
    third_channel_average = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))    #%%
    third_channel_average_flip = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))    #%%
    neurite_distances_averages = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))    #%%
    neurite_distances_averages_250max = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)))    #%%
    third_channel_average[index[:,0].astype(np.uint16),index[:,1].astype(np.uint16)] = np.asarray(means)[:,0]
    third_channel_average_flip[index_flip[:,0].astype(np.uint16),index_flip[:,1].astype(np.uint16)] = np.asarray(means_flip)[:,0]
    neurite_distances_averages[index_neurites[:,0].astype(np.uint16),index_neurites[:,1].astype(np.uint16)] = np.asarray(means_neurites)[:,0]
    neurite_distances_averages_250max[index_neurites_250[:,0].astype(np.uint16),index_neurites_250[:,1].astype(np.uint16)] = np.asarray(means_neurites_250)[:,0]
    third_channel_image_small = resize(third_channel_image,output_shape=[int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)], preserve_range=True)
    
    #%
    
    density_heatmaps[i] = out_heatmap
    third_channel_raw_heatmaps[i] = third_channel_average
    third_channel_flip_heatmaps[i] = third_channel_average_flip
    neurite_distance_heatmaps_counts_positive[i] = neurite_positive_heatmap
    neurite_distance_heatmaps_average_intensity[i] = neurite_distances_averages
    raw_pSYN_intensity_small[i] = third_channel_image_small
    raw_pSYN_intensity[i] = third_channel_image
    
    #%
   
    section_restricted_mask_large = resize(section_restricted_mask, [int(current_tile_SizeY)/19,int(current_tile_SizeX)/19], order=0, mode='constant',
                                           cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None).astype(np.uint8)
    
    section_restricted_mask_large_coords = np.where(section_restricted_mask_large==1)#.astype(np.uint16)
    
    section_restricted_mask_large_coords = np.hstack(((section_restricted_mask_large_coords[0]).reshape(len(section_restricted_mask_large_coords[0]),1),
                                                       (section_restricted_mask_large_coords[1]).reshape(len(section_restricted_mask_large_coords[0]),1))).astype(np.uint16)
    section_restricted_mask_large_coords = section_restricted_mask_large_coords / (scaling_factor /19)
    
    section_restricted_mask_large_coords = np.round(section_restricted_mask_large_coords).astype(np.uint16)
    section_restricted_mask_large_coords = np.hstack((section_restricted_mask_large_coords,np.ones((np.shape(section_restricted_mask_large_coords)[0],1)))).astype(np.uint16)
    section_restricted_mask_large_coords = pd.DataFrame(section_restricted_mask_large_coords)
    section_restricted_mask_counts = section_restricted_mask_large_coords.groupby([0,1]).count()
   
    section_restricted_mask_counts_index = section_restricted_mask_counts.index.values
    section_restricted_mask_counts_index = np.vstack(section_restricted_mask_counts_index).astype(np.uint16)
    
    
    section_restricted_mask_counts = np.hstack((section_restricted_mask_counts_index,np.asarray(section_restricted_mask_counts)))
    
    #
    ## keep quadrants with >90% max number pixels
    section_restricted_mask_counts = section_restricted_mask_counts[section_restricted_mask_counts[:,2]>int((scaling_factor /19)*(scaling_factor /19)*.90),:]
   
    
   
    full_tile_areas_segmentation = np.zeros((int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor))).astype(np.uint8) 
   
    section_restricted_mask_counts =  section_restricted_mask_counts[ section_restricted_mask_counts[:,0] >=0 ]
    section_restricted_mask_counts =  section_restricted_mask_counts[ section_restricted_mask_counts[:,0] <=(np.shape(full_tile_areas_segmentation)[0]-1) ]
    section_restricted_mask_counts =  section_restricted_mask_counts[ section_restricted_mask_counts[:,1] >=0 ]
    section_restricted_mask_counts =  section_restricted_mask_counts[ section_restricted_mask_counts[:,1] <=(np.shape(full_tile_areas_segmentation)[1]-1) ]
    section_restricted_mask_counts =  section_restricted_mask_counts.astype(np.uint16)
   
    
   
    
    full_tile_areas_segmentation[(section_restricted_mask_counts[:,0]).astype(np.uint16),(section_restricted_mask_counts[:,1]).astype(np.uint16)] = 1
    
    
    
    
    
    section_restricted_mask_full_tiles[i] = np.flipud(full_tile_areas_segmentation)
    #%
    reference_image = resize(reference_image, [np.shape(czi_metadata)[4]/scaling_factor,np.shape(czi_metadata)[5]/scaling_factor], order=0, mode='constant',
                                            cval=0, clip=True, preserve_range=True, anti_aliasing=None, 
                                            anti_aliasing_sigma=None)

    reference_image = np.flipud(reference_image)
    reference_image_storage[i] = reference_image
    
    current_tilescan_pixels= np.hstack((current_tile_SizeY,current_tile_SizeX))
    original_tilescan_pixels[i]=current_tilescan_pixels
    
    
    ##Atlas registration
    flatfile = input_file_key_data[i,5]
    registered_roi_mask = input_file_key_data[i,6]
    registered_roi_mask = AICSImage(registered_roi_mask) 
    registered_roi_mask = registered_roi_mask.get_image_dask_data("YX")  # returns out-of-memory 4D dask array
    registered_roi_mask = registered_roi_mask.compute()

    #%
    registered_roi_mask = np.where(registered_roi_mask==255,1,0)

    #%

    palette=False


    name_start ='".*"'
    #print(re.search(name_start, line)[0])
    roi_names = []
    import re
    palette=[]
    label_data = []
    with open(labelfile) as f:
        for line in f:
            lbl=re.match(r'\s*\d+\s+(\d+)\s+(\d+)\s+(\d+)',line)
            if lbl:
                palette.append((int(lbl[1]),int(lbl[2]),int(lbl[3])))
               # print(lbl[8])
                roi_names.append(re.search(name_start, line)[0])
                label_data.append(line)
    print(f"{len(palette)} labels parsed")
     
    import struct
    with open(flatfile,"rb") as f:
        b,w,h=struct.unpack(">BII",f.read(9))
        print(b)
        print(w)
        print(h)
        data=struct.unpack(">"+("xBH"[b]*(w*h)),f.read(b*w*h))
    print(f"{b} bytes per pixel, {w} x {h} resolution")
    roi_names = np.vstack(roi_names)
    ##Convert flat file to image of ROI by label.txt row identifier
    #%

    #%
    registered_atlas = np.asarray(data).reshape((h,w))
    registered_atlas_original = np.asarray(data).reshape((h,w))

    #%
    ##Resize registered atlas to photoshop overlsy
    registered_atlas = resize(registered_atlas, np.shape(registered_roi_mask), order=0, mode='constant',
                                            cval=0, clip=True, preserve_range=True, anti_aliasing=None, 
                                            anti_aliasing_sigma=None)


    #%
    ## Find where photoshop overlat selected tilescan specific atlas subset
    mask_lims = np.where(registered_roi_mask == 1)
    x_min = np.min(mask_lims[1])
    x_max = np.max(mask_lims[1])
    y_min = np.min(mask_lims[0])
    y_max = np.max(mask_lims[0])

    cropped_roi_mask = registered_atlas[y_min:y_max,x_min:x_max].astype(np.uint16)
    #%
    ##Resize cropped roi mask to tilescan size
    ## Previous [np.shape(czi_metadata)[4],np.shape(czi_metadata)[5]] includes zero spacing. Changed below for pure coords size
    cropped_roi_mask = resize(cropped_roi_mask,[int(current_tile_SizeY/scaling_factor),int(current_tile_SizeX/scaling_factor)], order=0, mode='constant',
                                            cval=0, clip=True, preserve_range=True, anti_aliasing=False, 
                                            anti_aliasing_sigma=None)
    
    cropped_roi_mask = np.flipud(cropped_roi_mask * full_tile_areas_segmentation)
    
    cropped_roi_masks[i] = cropped_roi_mask
    
 #%
#print(i)
set_loky_pickler('cloudpickle')    
Parallel(n_jobs=10, verbose=50, backend='threading')(delayed(generate_heatmaps)(i)for i in file_i_to_process)#range(0,np.shape(input_file_key_data)[0]))      
    





input_file_key_data_crops = np.asarray(dask.dataframe.read_csv(input_file_key_crops, delimiter=',', header=None)) 
input_file_key_data_header_crops = input_file_key_data_crops[0,:]
input_file_key_data_crops = (input_file_key_data_crops[file_i_to_process,:]).astype(np.uint16)

input_file_key_data = np.asarray(dask.dataframe.read_csv(input_file_key, delimiter=',', header=None)) 
input_file_key_data_header = input_file_key_data[0,:]
input_file_key_data = input_file_key_data[1::,:]



density_psyn_correlation = [None] * np.shape(input_file_key_data)[0]
density_psyn_correlation_manual = [None] * np.shape(input_file_key_data)[0]
density_psyn_correlation_2nd_motor = [None] * np.shape(input_file_key_data)[0]
density_psyn_correlation_2nd_motor_L5 = [None] * np.shape(input_file_key_data)[0]
density_psyn_correlation_L5_6_cingulate = [None] * np.shape(input_file_key_data)[0]
density_psyn_correlation_L5_cingulate = [None] * np.shape(input_file_key_data)[0]

density_psyn_correlation_L6 = [None] * np.shape(input_file_key_data)[0]
#density_psyn_correlation_L6_cingulate = [None] * np.shape(input_file_key_data)[0]
for row_i in file_i_to_process:#input_file_key_data_crops:
    #row_i = row_i - 1
    row_crops = np.ravel(input_file_key_data_crops[input_file_key_data_crops[:,0]==row_i,:])
    #plt.clf()
    i = row_crops[0] 
    box_top_left = [row_crops[1],row_crops[2] ]
    width = row_crops[4]
    height = row_crops[5]
    
    ## fix crop area for 1478 by tile analysis
    box_top_left = [int(row_crops[1]/ 2),int(row_crops[2]/ 2) ] 
    width = int(row_crops[4] / 2)
    height = int(row_crops[5] / 2)
    #height = (np.shape(reference_image)[0] - box_top_left [0])
    #box_top_left [1] = 0
    
    #print(i)
    reference_image = reference_image_storage[i]
    
    #reference_image = np.flipud(reference_image)
    
    out_heatmap = density_heatmaps[i]
    third_channel_average = third_channel_raw_heatmaps[i]
    full_tile_areas_segmentation_flip = section_restricted_mask_full_tiles[i]
    current_third_channel_raw = np.flipud(raw_pSYN_intensity_small[i])
    current_cropped_roi_mask = cropped_roi_masks[i]
    current_tile_id_map = tile_ids_v2[i]
    #print(np.shape(out_heatmap))
    #print(np.shape(third_channel_average))
    #print(np.shape(full_tile_areas_segmentation_flip))
    
    
   
    
    density=np.ravel((out_heatmap)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    third_channel_intensity=np.ravel((third_channel_average)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    neurite_counts=np.ravel((neurite_distance_heatmaps_counts_positive[i])[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    neurite_distance=np.ravel((neurite_distance_heatmaps_average_intensity[i])[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    current_third_channel_raw=np.ravel((current_third_channel_raw)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    current_cropped_roi_mask=np.ravel((current_cropped_roi_mask)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    full_tiles = np.ravel((full_tile_areas_segmentation_flip)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])
    current_tile_id_map = np.ravel((current_tile_id_map)[box_top_left[0]:box_top_left[0]+height,box_top_left[1]:box_top_left[1]+width])


    density = density[full_tiles==1]
    third_channel_intensity = third_channel_intensity[full_tiles==1]
    neurite_counts = neurite_counts[full_tiles==1]
    neurite_distance = neurite_distance[full_tiles==1]
    current_third_channel_raw = current_third_channel_raw[full_tiles==1]
    current_cropped_roi_mask = current_cropped_roi_mask[full_tiles==1]
    current_tile_id_map = current_tile_id_map[full_tiles==1]
    
    third_channel_cutoff_1 = np.quantile(third_channel_intensity,q=.66)
    third_channel_cutoff_2 = np.quantile(third_channel_intensity,q=1)
    
    data = np.hstack((density.reshape(len(density),1),third_channel_intensity.reshape(len(third_channel_intensity),1),
                      neurite_counts.reshape(len(neurite_counts),1),neurite_distance.reshape(len(neurite_distance),1),
                      current_third_channel_raw.reshape(len(current_third_channel_raw),1),
                      current_cropped_roi_mask.reshape(len(current_cropped_roi_mask),1),
                      current_tile_id_map.reshape(len(current_tile_id_map),1)))
    data_norm = np.max(data,axis=0)
    data_norm [5] = 1
    density_psyn_correlation[i] = data
    
    rows_28 = np.where(data[:,5] ==28,1,0)
    rows_29 = np.where(data[:,5] ==29,1,0)
    rows_230 = np.where(data[:,5] ==230,1,0)
    rows_236 = np.where(data[:,5] ==236,1,0)
    rows_231 = np.where(data[:,5] ==231,1,0)
    rows_237 = np.where(data[:,5] ==237,1,0)
    
    density_psyn_correlation_manual[i] = data[(np.ravel(rows_28 + rows_29 + rows_230 + rows_236 + rows_231 + rows_237)).astype(bool),:]
    
    secondary_motor_current =  data[(np.ravel(rows_28 + rows_29)).astype(bool),:]
    density_psyn_correlation_2nd_motor[i] = secondary_motor_current#data[(np.ravel(rows_28 + rows_29)).astype(bool),:]
    density_psyn_correlation_L5_6_cingulate[i] = data[(np.ravel(rows_230 + rows_236 + rows_231 + rows_237)).astype(bool),:]
    density_psyn_correlation_L5_cingulate[i] = data[(np.ravel(rows_230 + rows_236)).astype(bool),:]
    density_psyn_correlation_2nd_motor_L5[i] = data[(np.ravel(rows_28)).astype(bool),:]
    density_psyn_correlation_L6[i] = data[(np.ravel(rows_29 )).astype(bool),:]
    
    
    image_name = filenaming(i)
    
    
    fig, ax = plt.subplots()
    ax.imshow(reference_image)
    
    rectangle = Rectangle((box_top_left[1],box_top_left[0]), (width), (height), fc='none', ec="red")
    
    ax.add_patch(rectangle)
    fig.savefig(output_folder_name + image_name, format=image_format, dpi=1200)
    plt.show()




