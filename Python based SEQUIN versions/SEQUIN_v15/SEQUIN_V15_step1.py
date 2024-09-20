# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:33:53 2020

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
from joblib import dump, load
import lightgbm as lgb
import napari 


#%%
#########################################
## User defined required input parameters
#########################################



######################################################
## Pretrained LightGBM model and associated parameters
######################################################
## LightGBM model location
pretrained_LightGBM_model = 'C:/SEQUINv15_LightGBM_model_1.joblib'
## Trans-synaptic distance cutoff to use when selecting synaptic pairs. Units of micron. 0.5 = 500nm
synaptic_cutoff_classifier = .5
## Whether to use pretrained model
use_pretrained_classifier = False


#####################################################
## Set input/output folders and naming conventions
#####################################################
## Input folder
input_folder  = 'C:/Main_Experiment_Folder/'
input_folder = input_folder.replace('\\','/')
## Experimental image folder
image_folder = input_folder + 'Experimental_Image_folder/'#'X:/Active/Airycan2020/9-22-20 synapsin preconjugation testing/AS6/'#'Y:/Active/Airycan2020/9-28-20/stain3/AS6/'#
## Channel alignment image folder
chA_folder = input_folder + 'Channel_Alignment_Image_folder/'
## Main output folder
## Will contain multiple runs of the code if using different parameters
## User must create this folder.
outputfolder = input_folder + 'Master_output_folder/'#'X:/Active/Airycan2020/9-22-20 synapsin preconjugation testing/out/'#'Y:/Active/Airycan2020/9-28-20/stain3/out/'
## Name for run specific data output
out_name = "current_run_output_folder"


#############################################################
## Channel parameters. Spot assignments and intensity cutoffs
#############################################################
## If 'True' reads 3rd channel. If 'False' creates zero array for third channel as substitute
three_channel_image = True
## Define which raw iamge channel 0,1,2 correspong to PSD,SYN,3rd channel
sequin_reference = 1
sequin_partner = 2
third_channel = 0
## Post acquisition Z-intensity correction. Digital verson of laser ramping per SEQUIN protocol
z_intensity_correction_on = True
## Tope percentage of spots to analyze. 1 = 100%
percent_main = 1
## Cutoff IDs based on 1-Sequin reference, 2=sequin-partner
## Use precalculated background intensity cutoff if same value for all images is required
pre_caluculated_intensity_cutoff = False
image1_cutoff = 0
image2_cutoff = 0


#############################################################
## Third channel baseline subtraction
#############################################################
## Manually calculated value
manual_3rd_channel_bg_cutoff = 0
## Whether to perform third channel baseline substraction
third_channel_baseline_subtration = False


######################################################################################
## Third channel masking for synaptic loci distance measurements
## All masking based on third channel defined above
######################################################################################
## Whether to measure distance between synaptic loci and third channel mask
label_loci_to_mask = True
## User defined intensity cutoff. Requires understanding of the intensity of the channel to be masked. 
neurite_mask_intensity_cutoff = 110
## Quantile cutoff for masking. Requires understanding of the intensity of the channel to be masked. 
neurite_mask_gauss_quantile_cutoff = .0015
## Allow distance measurement to be variable. Allows for spots near image border to be interpretable.
max_neurite_distance_variable = True
## Max distance in microns to measure synaptic distance to masked third channel. Used when calculating variable radius 
max_radius = 2.0
## If max_neurite_distance_variable set to False, this value will be used.
max_neurite_distance = 2.0 ##um
## Sets max radius for neurite distance testing. Number of voxels times actual voxelsize must be greater than the distance cutoff.
neurite_field_voxel_radius = 52
## Set max cutoff distance to dilate third channel mask.
nm_cutoff_neurite = .2


###########################################################################################
## Parameters for measurement of user defined fields around synaptic partners and centroids
###########################################################################################
## Defines radius in voxels to create field of voxels around centroid for parsing 3rd channel metadata
## If 'Radius' * 'Voxelsize' > nm_radius then fields will be uniform spheres
synaptic_field_voxel_radius = 52 # 50 corresponds to 2um xy
## Define radius in micron to reduce size of field defined above
## If 'Radius' * 'Voxelsize' < nm_radius then fields will be non-uniform ellipsoids        
## Wether to measure fields around synaptic centroigs
process_variable_radii = True
## User specific radii to test. Values are in microns. Can be a single value
nm_cutoff_all = [.200]#,.400,.600,.800]#,1.000,1.400,1.800,2.200]#[.05] 


##################################
## Channel Alignment of raw images
##################################
## Wether to perform Channel alignment      
channel_align_image = True
## Top percent brightest spots to use for channel alignment calculation. 0.05 = Top 5%
percent_chA = 0.05
## Channel IDs correspond to the Raw original imaging data.
## Order of fluorophores in Channel Alignment image and Experimental data must be identical.
## Voxelsize between Channel Alignment image and Experimental data must be identical.
## Reference Channel alignment is measured from.
reference_channel_chA = 1
## Channels to align to reference
channels_to_align = [0,2]
## XY correction is measured highly unreliable. Recommend disregarding XY alignment. 
disregard_XY_alignment = True
## Override measurement of Channel Alignment Values and use User defined values.
## If 'channel_align_image' above set to False the no channel alignment will be performed even if 'channel_alignment_override' is set to True.
channel_alignment_override = False
## User defined channel alignment corrections for each channel
mannual_cha_ch0 = [2.16514106442566, 0.0, 0.0]
mannual_cha_ch1 = [0, 0.0, 0.0]
mannual_cha_ch2 = [-1.80336860507471, 0.0, 0.0]


####################################################################
## Maximum distance to measure trans-synaptic separation in microns. 
###################################################################
## Distance um for seleting NN pairs for extract metadata for. Reducing value will speed up processing at loss of non-synaptic spots metadata.
synaptic_cutoff = 3


##########################################################################################################
## Background subtraction values. Helpful for tilescans containing areas out of tissue or in White Matter.
##########################################################################################################
## Set WM baseline subtraction value if need
## wmbls1 corresponds to SEQUIN reference image. wmbl2 is SEQUIN partner image. wmbl3 in third channel for localization analysis
wmbls1 = 0
wmbls2 = 0
wmbls3 = 0



######################################################
## End of standard and required user input parameters.
######################################################



#%%
###############################################################
## Core code with Advanced parameters.
## Recommend just running, and not modifying, this block of code.
## Do not edit.
## Do no change advanced parameters 
##############################################################

###################################
## Advanced parameters
## Recommend leaving as deafults
##################################

## Filter spots by intensity of the other synaptic partner
cross_filter_spots = False
## Analyze the bottom percentage of spots with intensity filtering
bottom_percent = False
## Maximum and Minimum spots sizes for spot side filtering
max_spot_size = 10000000
min_spot_size = 7
## Set intensity background subtraction to highest frequency. Same as Imaris Baseline Subtraction.
use_hist_max_cutoff = False
## Deafult percentage intensity cutoff based upon intensity background subtraction of highest frequency. Similar to percentage of Imaris Baseline Subtraction.
## Was .1 9/19/2024. Reset to org processing.
percentile_intensity_cutoff = .02 
## Gaussian Filter image
gaussian_filter_image = False
## Gaussian sigma
gaussian = 0
## Auto calculate image specific third channel baseline subtraction value. 
third_channel_auto_BLS = False
##Quantile cutoff ran after 3rd channel BLS on adjusted image or raw image
quantile_filter_third_channel = False
quantile_cutoff_percent = 0.95
## currently not used ADS 5-13-2024
mask_3rd_channel_cutoff = 1000
## Apply rolling ball filter to image
rolling_ball_filtered = False
## Apply background subtraction prior to performing rolling bacll filter.
pre_rolling_ball_intensity_filter = False
pre_rb_wmbls1 = 0
pre_rb_wmbls2 = 0
## Auto calculatoin of Spot 1 and Spot 2 baseline subtraction values
auto_psd_syn_BLS = False
## Place holder channel alignment values. Do Not Change.
shift_raw_ch0 = [0,0,0]
shift_raw_ch1 = [0,0,0]
shift_raw_ch2 = [0,0,0]
## Now single images processed the same as Tilescans so leave as False.
tile_scan_980 = False
## Display testing of spot identification
display_detected_spots = False
## Tile ID to display
image_to_display = 0
## Terminate processing after displaying image
stop_after_display = True




############
## Core code
############


chA_files_names = np.sort(os.listdir(chA_folder))
chA_files = np.sort(os.listdir(chA_folder))
if len(chA_files) > 0:
    chA_files = np.asarray([x for x in chA_files if x.endswith('.czi')] )
    chA_files = pd.DataFrame(chA_files)
    chA_files = chA_files[chA_files != 'Thumbs.db']
    chA_files = chA_files.dropna()
    chA_files = np.asarray(chA_files)
    chA_files = np.ravel((chA_folder  + chA_files))
    load_chA_image = AICSImage(chA_files[0],reconstruct_mosaic=False)
    #chA_file = str(chA_filestr[0])
    #chA_file = chA_file[2:-2]
if len(chA_files) == 0:
    print('No cha files')


files = np.sort(os.listdir(image_folder))
files = np.asarray([x for x in files if x.endswith('.czi')] )

files = pd.DataFrame(files)
files = files[files != 'Thumbs.db']
files = files.dropna()
filestr = np.asarray(files)

  
file = str(filestr[0])
file = file[2:-2]

files = np.ravel((image_folder + files))


name_start ='scan(.*)_Out'

data_storage = []


load_image = []
load_chA_image =[]
load_image = AICSImage(files[0],reconstruct_mosaic=False)


czi_metadata = czi.CziFile(image_folder + '/' + file)

metadatadict_czi_True = czi_metadata.metadata(raw=True)
metadatadict_czi = czi_metadata.metadata(raw=False)


xy_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
z_dim = (metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
voxelsize=[z_dim,xy_dim,xy_dim]

flip=0



outfolder = out_name + "/"

path = os.path.join(outputfolder, outfolder)
ch1_path = os.path.join(path, 'ch1')
ch2_path = os.path.join(path, 'ch2')

## 9-19-2022 Confirm no needed!!!
#metadata = 0 #0=off, 1=on

if os.path.exists(path) == False:
    os.mkdir(path)
    
if os.path.exists(ch1_path) == False:
    os.mkdir(ch1_path)

if os.path.exists(ch2_path) == False:
    os.mkdir(ch2_path)
    
qi = []

if tile_scan_980 == True:
    number_tile_regions = len(metadatadict_czi['ImageDocument']
              ['Metadata']
              ['Experiment']
              ['ExperimentBlocks']
              ['AcquisitionBlock']
              ['SubDimensionSetups']
              ['RegionsSetup']
              ['SampleHolder']
              ['TileRegions']
              ['TileRegion'])
    
    tile_region_names = []
    
    
    for current_tile_name_i in range(0,number_tile_regions):
        current_tile_name = (metadatadict_czi['ImageDocument']
                  ['Metadata']
                  ['Experiment']
                  ['ExperimentBlocks']
                  ['AcquisitionBlock']
                  ['SubDimensionSetups']
                  ['RegionsSetup']
                  ['SampleHolder']
                  ['TileRegions']
                  ['TileRegion']
                  [current_tile_name_i]
                  ['Name'])
        tile_region_names.append(current_tile_name)
    tile_region_names = np.vstack(tile_region_names)
## Currently not used
## Hardcoded to need variables but should be removed
spot_1_channel = 1 ##Typically PSD for SEQUIN analysis
spot_2_channel = 2 ##Typically SYN for SEQUIN
spot_3_channel = 0 ##3rd channel of interest 


#%


if use_pretrained_classifier == True:
    model_reload = load(pretrained_LightGBM_model) 




def  filename_slicing(file):
    #file_name = file[0:-4]
    file_name = file[0:16] + file[63::]
    return file_name


def identify_channels(image_ch0, image_ch1, image_ch2):

    if sequin_reference == 0:
        image1 = image_ch0 ##SEQUIN reference
    if sequin_reference == 1:
        image1 = image_ch1 ##SEQUIN reference    
    if sequin_reference == 2:
        image1 = image_ch2 ##SEQUIN reference    
   
    if  sequin_partner == 0:
        image2 = image_ch0 ##SEQUIN reference
    if  sequin_partner == 1:
        image2 = image_ch1 ##SEQUIN reference    
    if  sequin_partner == 2:
        image2 = image_ch2 ##SEQUIN reference        
   
    
    if third_channel == 0:
        image3 = image_ch0 ##SEQUIN reference
    if third_channel == 1:
        image3 = image_ch1 ##SEQUIN reference    
    if third_channel == 2:
        image3 = image_ch2 ##SEQUIN reference       
   
   
 
    return image1, image2, image3

    
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
   
    return image_z_corrected#, image_z_median, image_zcorrected_median

def select_channels(image,shift_raw_ch0,shift_raw_ch1,shift_raw_ch2,spot_1_channel,spot_2_channel,spot_3_channel,current_channel_to_align,reference_channel_chA):
    image1 = []
    image2 = []
    image3 = []
    
    if z_intensity_correction_on == True:
        print('z correcting')
        for channel_number in range(0,np.shape(image)[0]):
            z_corrected_image = z_intensity_correction(image[channel_number,:,:,:])
            image[channel_number,:,:,:] = z_corrected_image
    else:
        print('not correcting')
    
    if chA_image == True:
        image1 = image[reference_channel_chA,:,:,:]
        image2 = image[current_channel_to_align,:,:,:]
        image3 = np.zeros(np.shape(image1))
    
            
            
    if channel_alignment_override == True:
        shift_raw_ch0 = mannual_cha_ch0
        shift_raw_ch1 = mannual_cha_ch1
        shift_raw_ch2 = mannual_cha_ch2
    
    if disregard_XY_alignment == True:
        shift_raw_ch0[1] = 0
        shift_raw_ch0[2] = 0
        shift_raw_ch1[1] = 0
        shift_raw_ch1[2] = 0       
        
        
        
        
    if (channel_align_image == True) and (chA_calculated == True):
        image_ch0 = image[0,:,:,:] 
        image_ch1 = image[1,:,:,:]
        if three_channel_image == True:
            image_ch2 = image[2,:,:,:]
        if three_channel_image == False:
            image_ch2 = np.zeros(np.shape(image_ch0))

            
        
        image_ch0 = scipy.ndimage.shift(image_ch0, shift=shift_raw_ch0,
                                       order=3, mode='constant', cval=0.0, prefilter=False)
    
        
        image_ch1 = scipy.ndimage.shift(image_ch1, shift=shift_raw_ch1,
                                       order=3, mode='constant', cval=0.0, prefilter=False)


        image_ch2 = scipy.ndimage.shift(image_ch2, shift=shift_raw_ch2,
                                       order=3, mode='constant', cval=0.0, prefilter=False)

        image1, image2, image3 = identify_channels(image_ch0, image_ch1, image_ch2)
        if gaussian_filter_image == True:
            image1 = filters.gaussian(image1, sigma=gaussian)
            image2 = filters.gaussian(image2, sigma=gaussian)
            image3 = filters.gaussian(image3, sigma=gaussian)
        
        all_shifts = [shift_raw_ch0,shift_raw_ch1,shift_raw_ch2]
        for current_shift in all_shifts:
            shift = np.asarray(current_shift)
            shift = abs(shift.astype(int))
            edge_crop = shift + 1 
            image_shape_crop = np.shape(image1)
            #Z crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.
            if current_shift[0] > 0:
                image1 = image1[edge_crop[0]::,:,:]
                image2 = image2[edge_crop[0]::,:,:]
                image3 = image3[edge_crop[0]::,:,:]
            elif current_shift[0] < 0:
                image1 = image1[0:(image_shape_crop[0]-edge_crop[0]),:,:]
                image2 = image2[0:(image_shape_crop[0]-edge_crop[0]),:,:]
                image3 = image3[0:(image_shape_crop[0]-edge_crop[0]),:,:]
                
            #Y crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
            if current_shift[1] > 0:
                image1 = image1[:,edge_crop[1]::,:]
                image2 = image2[:,edge_crop[1]::,:]
                image3 = image3[:,edge_crop[1]::,:]
            elif current_shift[1] < 0:
                image1 = image1[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
                image2 = image2[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
                image3 = image3[:,0:(image_shape_crop[1]-edge_crop[1]),:] 
                
                
            #X crop Position shifts leave empty pixels at beginning of demenison. Negative values leave empty pixels at end of dimension.   
            if current_shift[2] > 0:
                image1 = image1[:,:,edge_crop[2]::]
                image2 = image2[:,:,edge_crop[2]::]
                image3 = image3[:,:,edge_crop[2]::]
            elif current_shift[2] < 0:
                image1 = image1[:,:,0:(image_shape_crop[2]-edge_crop[2])]
                image2 = image2[:,:,0:(image_shape_crop[2]-edge_crop[2])] 
                image3 = image3[:,:,0:(image_shape_crop[2]-edge_crop[2])] 

    if (channel_align_image == False):
        image_ch0 = image[0,:,:,:] 
        image_ch1 = image[1,:,:,:] 
        image_ch2 = image[2,:,:,:] 
      
        image1, image2, image3 = identify_channels(image_ch0, image_ch1, image_ch2)
       
    third_channel_bg_cutoff = 0
    if (third_channel_baseline_subtration == True) and (chA_calculated == True):
        third_channel_bg_cutoff = manual_3rd_channel_bg_cutoff
        if third_channel_auto_BLS == True:
            histogram, bin_edges = np.histogram(image3, bins=1000)#, range=(0, 1))
            histmax = np.max(histogram)
            hist_max_index = int(np.asarray((np.where(histogram == histmax))))
            hist_max_cutoff = ((bin_edges[hist_max_index]) + (bin_edges[hist_max_index + 1])) /2
            third_channel_bg_cutoff = np.max((hist_max_cutoff,manual_3rd_channel_bg_cutoff))
        
        
        image3 = np.where(image3>third_channel_bg_cutoff,image3 -third_channel_bg_cutoff,0)
        image3 = image3.astype(np.uint16)
        
    if (quantile_filter_third_channel == True) and (chA_calculated == True):
        quantile_cutoff_value = np.quantile(image3,q=quantile_cutoff_percent)
        mask = np.where(image3>quantile_cutoff_value,1,0)
        ball = morphology.ball(12)
        binarized_dilated = np.invert(morphology.binary_dilation(mask,ball))
        image3 = image3 *  binarized_dilated
        
        
        
    if (auto_psd_syn_BLS == True):
        histogram, bin_edges = np.histogram(image2, bins=1000)#, range=(0, 1))
        histmax = np.max(histogram)
        hist_max_index = int(np.asarray((np.where(histogram == histmax))))
        hist_max_cutoff = ((bin_edges[hist_max_index]) + (bin_edges[hist_max_index + 1])) /2
        image2 = np.where(image2>hist_max_cutoff,image2 -hist_max_cutoff,0)
        image2 = image2.astype(np.uint16)
        
        histogram, bin_edges = np.histogram(image1, bins=1000)#, range=(0, 1))
        histmax = np.max(histogram)
        hist_max_index = int(np.asarray((np.where(histogram == histmax))))
        hist_max_cutoff = ((bin_edges[hist_max_index]) + (bin_edges[hist_max_index + 1])) /2
        image1 = np.where(image1>hist_max_cutoff,image1 -hist_max_cutoff,0)
        image1 = image1.astype(np.uint16) 
        
    
    third_channel_bg_cutoff = int(third_channel_bg_cutoff)

    if pre_rolling_ball_intensity_filter == True:
        image1 = np.where(image1>pre_rb_wmbls1,image1-pre_rb_wmbls1,0)
        image2 = np.where(image2>pre_rb_wmbls2,image2-pre_rb_wmbls2,0)
    
    
    if rolling_ball_filtered == True:
        image1 = image1 - (restoration.rolling_ball(image1,radius=1))
        image2 = image2 - (restoration.rolling_ball(image2,radius=1))
        
    save_df_variables = pd.DataFrame(dir())
    save_df_variables.to_csv(path + '_' + '_selectChannels_variables.txt',header=False, index=False)
    
    
    del current_channel_to_align
    del image
    del reference_channel_chA
    del spot_1_channel
    del spot_2_channel
    del spot_3_channel
    gc.collect()
    
    save_df_variables = pd.DataFrame(dir())
    save_df_variables.to_csv(path + '_' + '_selectChannels_variables.txt',header=False, index=False)
    
    return image1, image2, image3,third_channel_bg_cutoff
    

tic = time.perf_counter()


def get_metadata (label_image, intensity_image, intensity_image_other,intensity_image_3rd_channel,synaptic_loci,psd_syn_i,psd_spots_image, syn_spots_image,dilated_mask):
#%  
    dilated_mask = np.where(dilated_mask == 0,1,0)
    intensity_image_3rd_channel_mask_removed = intensity_image_3rd_channel * dilated_mask
    
    #Extract metadata based upon raw unprocessed image spots were created on
    props = ('label','area','mean_intensity','weighted_centroid','min_intensity',
             'max_intensity','major_axis_length','minor_axis_length','centroid',
             'equivalent_diameter','euler_number','extent','filled_area',
             'inertia_tensor_eigvals','convex_area','solidity')
    
    table1 = measure.regionprops_table(label_image=label_image, intensity_image=intensity_image,properties=props)
    table1 = np.asarray(pd.DataFrame.from_dict(table1, orient='columns'))
    
      
   
    #Extract metadata based upon raw unprocessed image of other channel
    props = ('mean_intensity','min_intensity','max_intensity')
    table2 = measure.regionprops_table(label_image=label_image, intensity_image=intensity_image_other,properties=props) 
    table2 = np.asarray(pd.DataFrame.from_dict(table2, orient='columns'))

    props = ('mean_intensity','min_intensity','max_intensity')
    table3 = measure.regionprops_table(label_image=label_image, intensity_image=intensity_image_3rd_channel,properties=props) 
    table3 = np.asarray(pd.DataFrame.from_dict(table3, orient='columns'))
    
    props = ('mean_intensity','min_intensity','max_intensity')
    table4 = measure.regionprops_table(label_image=label_image, intensity_image=intensity_image_3rd_channel_mask_removed,properties=props) 
    table4 = np.asarray(pd.DataFrame.from_dict(table3, orient='columns'))
    
    properties = measure.regionprops(label_image=label_image, intensity_image=intensity_image)# 
   
                 
      
    inertia_tensor_table = []
    moments_table = []
    moments_central_table = []
    moments_normalized_table = []
    weighted_moments_table = []
    weighted_moments_central_table = []
    weighted_moments_normalized_table = []
    
    entropy_all = []
    bases = [2,10,np.e]
    skew_kurtosis_all = []
      
      
      
   
    for row in properties:
        inertia_tensor = np.ravel(row.inertia_tensor).reshape(9,1).T
        inertia_tensor_table.append(inertia_tensor) 
        
        moments = np.ravel(row.moments).reshape(64,1).T
        moments_table.append(moments)
        
        moments_central = np.ravel(row.moments_central).reshape(64,1).T
        moments_central_table.append(moments_central)
        
        moments_normalized = np.ravel(row.moments_normalized).reshape(64,1).T
        moments_normalized_table.append(moments_normalized)
        
        weighted_moments = np.ravel(row.moments_weighted).reshape(64,1).T
        weighted_moments_table.append(weighted_moments)
        
        weighted_moments_central = np.ravel(row.moments_weighted_central).reshape(64,1).T
        weighted_moments_central_table.append(weighted_moments_central)
        
        weighted_moments_normalized = np.ravel(row.moments_weighted_normalized).reshape(64,1).T
        weighted_moments_normalized_table.append(weighted_moments_normalized)
  
        
        intensity_image1 = row.image_intensity
        intensity_image2 = np.where(intensity_image1 > 0 , intensity_image1 , np.nan)
        label = row.label
       
        entropy_current = []
        for base_value in bases:
            entropy = measure.shannon_entropy(intensity_image2, base=base_value)
            entropy_current = np.hstack([entropy_current,entropy])
        entropy_all.append(entropy_current)
            
        #intensity_image = row.image_intensity
        intensity_image_ravel = np.ravel(intensity_image1)
        intensity_image_ravel = intensity_image_ravel[np.nonzero(intensity_image_ravel)]
        skew_current = scipy.stats.skew(intensity_image_ravel)
        kurtosis_current = scipy.stats.kurtosis(intensity_image_ravel)
        std_current = np.std(intensity_image_ravel)#.reshape(1,1)
        mean_current = np.mean(intensity_image_ravel)
        cv_current =  std_current /  mean_current * 100
        
        median_current = np.median(intensity_image_ravel)
        std_median_current = std_current / median_current
        
        
        skew_kurtosis = np.hstack([skew_current, kurtosis_current,std_current,cv_current,median_current,std_median_current])
        skew_kurtosis_all.append(skew_kurtosis) 
        
         
    inertia_tensor_table=np.stack(inertia_tensor_table,axis=1)
    moments_table=np.stack(moments_table,axis=1)
    moments_central_table=np.stack(moments_central_table,axis=1)
    moments_normalized_table=np.stack(moments_normalized_table,axis=1)
    weighted_moments_table=np.stack(weighted_moments_table,axis=1)
    weighted_moments_central_table=np.stack(weighted_moments_central_table,axis=1)
    weighted_moments_normalized_table=np.stack(weighted_moments_normalized_table,axis=1)
    
    inertia_tensor_table = inertia_tensor_table[0,:,:]
    moments_table = moments_table[0,:,:]
    moments_central_table = moments_central_table[0,:,:]
    moments_normalized_table = moments_normalized_table[0,:,:]
    weighted_moments_table = weighted_moments_table[0,:,:]
    weighted_moments_central_table = weighted_moments_central_table[0,:,:]
    weighted_moments_normalized_table = weighted_moments_normalized_table[0,:,:]
    
    entropy_all = np.asarray(entropy_all)
    skew_kurtosis_all = np.asarray(skew_kurtosis_all) 
    

    
                    
                       
    
    all_data2 = np.hstack([table1, table2, table3,table4,inertia_tensor_table, moments_table, moments_central_table, moments_normalized_table, weighted_moments_table, weighted_moments_central_table, weighted_moments_normalized_table,entropy_all,skew_kurtosis_all]) #skew_kurtosis_all
    del inertia_tensor_table 
    del moments_table 
    del moments_central_table 
    del moments_normalized_table 
    del weighted_moments_table 
    del weighted_moments_central_table 
    del weighted_moments_normalized_table
    
    del weighted_moments
    del weighted_moments_central
    del weighted_moments_normalized
    del table1
    del table2
    del skew_kurtosis_all
    del entropy_all
    del inertia_tensor
    
    del label_image
    del intensity_image
    del intensity_image_other
    del intensity_image1
    del intensity_image2
    del intensity_image_ravel
    
    del moments
    del moments_central
    del moments_normalized
    
    if psd_syn_i == 0:
        if (process_variable_radii == False) or (running_channel_alignment == True):
            all_3rd_channel_intensity_data = []
            all_3rd_channel_intensity_data_flipped = []
        
        if (process_variable_radii == True) and (running_channel_alignment == False):
            image_dims = np.shape(intensity_image_3rd_channel)
            intensity_image_3rd_channel_flipped = np.flip(intensity_image_3rd_channel)
            intensity_image_3rd_channel_mask_removed_flipped = np.flip(intensity_image_3rd_channel_mask_removed)
           
            for reference_type in range(0,3):
                if reference_type == 0:
                    synaptic_centroids = np.asarray([(synaptic_loci[:,2]+synaptic_loci[:,10])/2,
                                                     (synaptic_loci[:,3]+synaptic_loci[:,11])/2,
                                                     (synaptic_loci[:,4]+synaptic_loci[:,12])/2]).T.astype(float)
                    reference_type_name = 'synaptic_centroid'
                
                if reference_type == 1:
                    synaptic_centroids = synaptic_loci[:,2:5]
                    reference_type_name = 'psd_centroid'
    
    
                if reference_type == 2:
                    synaptic_centroids = synaptic_loci[:,10:13]
                    reference_type_name = 'syn_centroid'
    
                 
                for nm_cutoff in nm_cutoff_all:
                    
                    xy_radius_scale = 1
                    
                    radius = synaptic_field_voxel_radius                                                            
                    
                    radius_check = synaptic_field_voxel_radius ##If manuall switch to radius on line above
                    
                    # Create integers for coordinate adjustement to set maximum possible spot size
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
                        
                        ###Choice of keep determines XYZ balance of voxels. nm --> sphere, radius--> ellipse
                        ### Consider making dynamic at user input parameter steps
                        ## 8/2/2022 changes to keep from keep_nm
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
                            third_channel_voxel_intensity = intensity_image_3rd_channel[z_indices,y_indices,x_indices]
                            third_channel_voxel_intensity2 = np.vstack((third_channel_voxel_intensity,third_channel_voxel_intensity,third_channel_voxel_intensity)).T
                           
                            third_channel_voxel_intensity_flipped = intensity_image_3rd_channel_flipped[z_indices,y_indices,x_indices]
                            third_channel_voxel_intensity_flipped2 = np.vstack((third_channel_voxel_intensity_flipped,third_channel_voxel_intensity_flipped,third_channel_voxel_intensity_flipped)).T
                           
                             
                            third_channel_voxel_intensity_mask_removed = intensity_image_3rd_channel_mask_removed[z_indices,y_indices,x_indices]
                            third_channel_voxel_intensity_mask_removed2 = np.vstack((third_channel_voxel_intensity_mask_removed,
                                                                                     third_channel_voxel_intensity_mask_removed,
                                                                                     third_channel_voxel_intensity_mask_removed)).T
                          
                            third_channel_voxel_intensity_mask_removed_flipped = intensity_image_3rd_channel_mask_removed_flipped[z_indices,y_indices,x_indices]
                            third_channel_voxel_intensity_mask_removed_flipped2 = np.vstack((third_channel_voxel_intensity_mask_removed_flipped,
                                                                                             third_channel_voxel_intensity_mask_removed_flipped,
                                                                                             third_channel_voxel_intensity_mask_removed_flipped)).T
                          
                            
                           
                            
                            weighted_zyx = (np.sum(zyx_indices * third_channel_voxel_intensity2,axis=0)) / np.sum(third_channel_voxel_intensity)
                            weighted_zyx_flip = (np.sum(zyx_indices * third_channel_voxel_intensity_flipped2,axis=0)) / np.sum(third_channel_voxel_intensity_flipped)
                          
                            weighted_zyx_mask_removed = (np.sum(zyx_indices * third_channel_voxel_intensity_mask_removed2,axis=0)) / np.sum(third_channel_voxel_intensity_mask_removed)
                            weighted_zyx_mask_removed_flip = (np.sum(zyx_indices * third_channel_voxel_intensity_mask_removed_flipped2,axis=0)) / np.sum(third_channel_voxel_intensity_mask_removed_flipped)
                          
                            
                            synaptic_field_coords = (weighted_zyx * voxelsize).reshape(1,3)
                            synaptic_field_coords_flip = (weighted_zyx_flip * voxelsize).reshape(1,3)
                            
                            synaptic_field_coords_mask_removed = (weighted_zyx_mask_removed * voxelsize).reshape(1,3)
                            synaptic_field_coords_mask_removed_flip = (weighted_zyx_mask_removed_flip * voxelsize).reshape(1,3)
                            
                              
                            current_psd_coords = (synaptic_loci[row_i,2:5].astype(float) * voxelsize).reshape(1,3)
                            current_synapsin_coords = (synaptic_loci[row_i,10:13].astype(float) * voxelsize).reshape(1,3)
                            
                            psd_to_field_weighted_centroid = scipy.spatial.distance.cdist(synaptic_field_coords, current_psd_coords, metric='euclidean')
                            syn_to_field_weighted_centroid = scipy.spatial.distance.cdist(synaptic_field_coords, current_synapsin_coords, metric='euclidean')
                            
                            psd_to_field_weighted_centroid_flip = scipy.spatial.distance.cdist(synaptic_field_coords_flip, current_psd_coords, metric='euclidean')
                            syn_to_field_weighted_centroid_flip = scipy.spatial.distance.cdist(synaptic_field_coords_flip, current_synapsin_coords, metric='euclidean')
                            
                            psd_to_field_weighted_centroid_mask_removed = scipy.spatial.distance.cdist(synaptic_field_coords_mask_removed, current_psd_coords, metric='euclidean')
                            syn_to_field_weighted_centroid_mask_removed = scipy.spatial.distance.cdist(synaptic_field_coords_mask_removed, current_synapsin_coords, metric='euclidean')
                            
                            psd_to_field_weighted_centroid_mask_removed_flip = scipy.spatial.distance.cdist(synaptic_field_coords_mask_removed_flip, current_psd_coords, metric='euclidean')
                            syn_to_field_weighted_centroid_mask_removed_flip = scipy.spatial.distance.cdist(synaptic_field_coords_mask_removed_flip, current_synapsin_coords, metric='euclidean')
                          
                            
                            
                            nn_confirmation = scipy.spatial.distance.cdist(current_psd_coords, current_synapsin_coords, metric='euclidean')
                             
                            synaptic_field_data = np.ravel(np.hstack((
                                synaptic_field_coords,
                                current_psd_coords,
                                current_synapsin_coords,
                                psd_to_field_weighted_centroid,
                                syn_to_field_weighted_centroid,
                                psd_to_field_weighted_centroid_flip,
                                syn_to_field_weighted_centroid_flip,
                                psd_to_field_weighted_centroid_mask_removed,
                                syn_to_field_weighted_centroid_mask_removed,
                                psd_to_field_weighted_centroid_mask_removed_flip,
                                syn_to_field_weighted_centroid_mask_removed_flip,
                                nn_confirmation)))
                            
                            third_channel_intensity_data = np.hstack((
                                number_synaptic_voxels,
                                np.mean(third_channel_voxel_intensity),
                                np.median(third_channel_voxel_intensity),
                                np.min(third_channel_voxel_intensity),
                                np.max(third_channel_voxel_intensity),
                                np.std(third_channel_voxel_intensity),
                                np.mean(third_channel_voxel_intensity_mask_removed),
                                np.median(third_channel_voxel_intensity_mask_removed),
                                np.min(third_channel_voxel_intensity_mask_removed),
                                np.max(third_channel_voxel_intensity_mask_removed),
                                np.std(third_channel_voxel_intensity_mask_removed))).reshape(1,11)
                            all_3rd_channel_intensity_data.append(third_channel_intensity_data)
                            
                            third_channel_intensity_data_flipped = np.hstack((
                                number_synaptic_voxels,
                                np.mean(third_channel_voxel_intensity_flipped),
                                np.median(third_channel_voxel_intensity_flipped),
                                np.min(third_channel_voxel_intensity_flipped),
                                np.max(third_channel_voxel_intensity_flipped),
                                np.std(third_channel_voxel_intensity_flipped),
                                np.mean(third_channel_voxel_intensity_mask_removed_flipped),
                                np.median(third_channel_voxel_intensity_mask_removed_flipped),
                                np.min(third_channel_voxel_intensity_mask_removed_flipped),
                                np.max(third_channel_voxel_intensity_mask_removed_flipped),
                                np.std(third_channel_voxel_intensity_mask_removed_flipped),
                                synaptic_field_data)).reshape(1,29)
                            all_3rd_channel_intensity_data_flipped.append(third_channel_intensity_data_flipped)                                                                  
                        elif number_synaptic_voxels <= 3:
                            third_channel_intensity_data = np.zeros((1,11))
                            third_channel_intensity_data[:] = np.nan
                            
                            third_channel_intensity_data_flipped = np.zeros((1,29))
                            third_channel_intensity_data_flipped[:] = np.nan
                            
                            all_3rd_channel_intensity_data.append(third_channel_intensity_data)
                            all_3rd_channel_intensity_data_flipped.append(third_channel_intensity_data_flipped)  
                            
                            
                    all_3rd_channel_intensity_data = np.vstack(all_3rd_channel_intensity_data)
                    all_3rd_channel_intensity_data_flipped = np.vstack(all_3rd_channel_intensity_data_flipped)
                    
                    column_headings = ['# voxels', 'mean', 'median','min','max','std','mean_mask_removed', 'median_mask_removed','min_mask_removed','max_mask_removed','std_mask_removed'
                                       ]                               
                    current_radius = np.repeat(radius, 11).reshape(1,11)
                    current_nm_cutoff = np.repeat(nm_cutoff, 11).reshape(1,11)
        
                    raw_label =  np.repeat('raw', 11).reshape(1,11)
                    reference_type_name_raw =  np.repeat(reference_type_name, 11).reshape(1,11)
                    all_3rd_channel_intensity_data = np.vstack((raw_label, reference_type_name_raw, current_radius, current_nm_cutoff, column_headings, all_3rd_channel_intensity_data))
                    
                    column_headings = ['# voxels', 'mean', 'median','min','max','std',
                                       'mean_mask_removed', 'median_mask_removed','min_mask_removed','max_mask_removed','std_mask_removed',
                                       'fieldZ','fieldY','fieldX','PSDZ','PSDY','PSDX','SYNZ','SYNY','SYNX','PSD-field-NN','SYN-field-NN',
                                       'Flipped_PSD-field-NN','Flipped_SYN-field-NN',
                                       'PSD-field-NN_mask_removed','SYN-field-NN_mask_removed','Flipped_PSD-field-NN_mask_removed','Flipped_SYN-field-NN_mask_removed',
                                       'PSD-SYN-NN']
                    current_radius = np.repeat(radius, 29).reshape(1,29)
                    current_nm_cutoff = np.repeat(nm_cutoff, 29).reshape(1,29)
                    flip_label =  np.repeat('flipped', 29).reshape(1,29)
                    reference_type_name_flip =  np.repeat(reference_type_name, 29).reshape(1,29)
                    
                    
                    
                    
                    all_3rd_channel_intensity_data_flipped = np.vstack((flip_label, reference_type_name_flip, current_radius, current_nm_cutoff, column_headings, all_3rd_channel_intensity_data_flipped))          
                    if (nm_cutoff == nm_cutoff_all[0]) and (reference_type == 0):
                        all_3rd_channel_intensity_data_all_radii = all_3rd_channel_intensity_data
                        all_3rd_channel_intensity_data_flipped_all_radii = all_3rd_channel_intensity_data_flipped               
                    else:
                        all_3rd_channel_intensity_data_all_radii = np.hstack((all_3rd_channel_intensity_data_all_radii,all_3rd_channel_intensity_data))
                        all_3rd_channel_intensity_data_flipped_all_radii = np.hstack((all_3rd_channel_intensity_data_flipped_all_radii,all_3rd_channel_intensity_data_flipped))                                      
                        
                    
                
            synaptic_loci_NN = synaptic_loci[:,[0,8,7,15,16]].reshape(np.shape(synaptic_loci)[0],5)
            zero_array = np.zeros((4,5))
            id_nn_titles = ['PSD_ID','SYN_ID','NN','NeuriteDistance','MaxNeuriteRadii']
            synaptic_loci_NN = np.vstack((zero_array,id_nn_titles,synaptic_loci_NN))
            
            all_3rd_channel_intensity_data = np.hstack((synaptic_loci_NN, all_3rd_channel_intensity_data_all_radii))
            all_3rd_channel_intensity_data_flipped = all_3rd_channel_intensity_data_flipped_all_radii
            
            
        psd_properties = measure.regionprops(label_image=psd_spots_image)# 
        syn_properties = measure.regionprops(label_image=syn_spots_image)# 
        psd_label_ids = []
        syn_label_ids = []
        for row in psd_properties:
            row_label = row.label
            psd_label_ids.append(row_label)
        for row in syn_properties:
            row_label = row.label
            syn_label_ids.append(row_label)


        psd_label_ids = np.vstack(psd_label_ids)
        syn_label_ids = np.vstack(syn_label_ids)
        all_rows_coords = []
        for row in synaptic_loci:
            current_psd_id = row[0]
            current_syn_id = row[8]
            psd_prop_row = int(np.where(psd_label_ids == current_psd_id)[0])
            syn_prop_row = int(np.where(syn_label_ids == current_syn_id)[0])
            psd_coords = psd_properties[psd_prop_row].coords
            syn_coords = syn_properties[syn_prop_row].coords
            matched_row = []
            for coords_row in psd_coords:
                matched0 = np.isin(syn_coords[:,0],coords_row[0])
                matched1 = np.isin(syn_coords[:,1],coords_row[1])
                matched2 = np.isin(syn_coords[:,2],coords_row[2])
                matched = np.hstack((matched0.reshape(len(matched0),1),matched1.reshape(len(matched0),1),matched2.reshape(len(matched0),1)))
                
                matched_sum = np.sum(matched,axis=1)
                if np.max(matched_sum) == 3:
                    matched_row.append(1)
                if np.max(matched_sum) != 3:
                    matched_row.append(0)
            total_psd = len(psd_coords)
            total_syn = len(syn_coords)
            total_matched = np.sum(matched_row)
            percent_psd = total_matched / total_psd * 100
            percent_syn = total_matched / total_syn * 100
            all_coords_data = np.hstack((current_psd_id,current_syn_id,total_psd, total_syn, total_matched, percent_psd, percent_syn))
            all_rows_coords.append( all_coords_data)
        all_rows_coords = np.vstack(all_rows_coords)
        
        
        
        
        
        

        
        del base_value
        del bases
        del cv_current
        del entropy
        del entropy_current
        del intensity_image_3rd_channel
        del kurtosis_current
        del label
        del mean_current
        del median_current
        del properties
        del props
        del psd_spots_image
        del psd_syn_i
        del row
        del skew_current
        del skew_kurtosis
        del std_current
        del std_median_current
        del syn_spots_image
        del synaptic_loci
        del table3
        gc.collect()
        save_df_variables = pd.DataFrame(dir())
        save_df_variables.to_csv(path + '_' + '_getMetaData1_variables.txt',header=False, index=False)
        
        return all_data2, all_3rd_channel_intensity_data, all_3rd_channel_intensity_data_flipped,all_rows_coords
#%  
    if psd_syn_i == 1:
       
        #del all_data2
        del base_value
        del bases
        del cv_current
        del entropy
        del entropy_current
        del intensity_image_3rd_channel
        del kurtosis_current
        del label
        del mean_current
        del median_current
        del properties
        del props
        del psd_spots_image
        del psd_syn_i
        del row
        del skew_current
        del skew_kurtosis
        del std_current
        del std_median_current
        del syn_spots_image
        del synaptic_loci
        del table3
        gc.collect()
        save_df_variables = pd.DataFrame(dir())
        save_df_variables.to_csv(path + '_' + '_getMetaData2_variables.txt',header=False, index=False)
        
        return all_data2





def spots_detection(image1, image2,file_name,out_name,path,i4,qi,file_name2,image3,tile_number,mosaic_tile_i,third_channel_bg_cutoff):
    #%
    for i in range(0,2):
         #%
        #i = 0
        #gaussian = 1
        if i == 0:
            #image = image1
            image = image1 
            if pre_caluculated_intensity_cutoff == True:
                cutoff = image1_cutoff
                
            
        if i == 1:
            #image = image2
            image = image2 
            if pre_caluculated_intensity_cutoff == True:
                cutoff = image2_cutoff
                
            
        if np.max(image) == 0:
            file_name2 =   re.split('/',file_name2)
            file_name2 = file_name2[len(file_name2)-1]     
            all_data_variable = np.zeros((1,856))
            Normal_Freq = np.zeros((150,1))
            psd_meta_data = np.zeros((1,856))
            syn_meta_data = np.zeros((1,856))
            synaptic_loci = np.zeros((1,856))
            current_df = np.zeros((1,856))
            file_name2 =   re.split('/',file_name2)
            file_name2 = file_name2[len(file_name2)-1]     
            np.savetxt(path + "/" + out_name   + "_" + file_name2 + "_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_Max0.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
             
            return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
         
        tic = time.perf_counter()
        maxi = np.max(image)
        
        
       
        image_norm = image
        
        
        
        
        if (pre_caluculated_intensity_cutoff == False) or (chA_calculated == False):
            image_norm = image / maxi
            histogram, bin_edges = np.histogram(image_norm, bins=1000, range=(0, 1))
            histmax = np.max(histogram)
    
            mu, sigma = scipy.stats.norm.fit(image_norm)
            best_fit_line = scipy.stats.norm.pdf(bin_edges, mu, sigma)
    
                  
            
            best_fit_max = np.max(best_fit_line) #np.quantile(best_fit_line, q = .9)
      
            best_fit_max_index = np.asarray((np.where(best_fit_line == best_fit_max)))
            best_fit_max_index = int(best_fit_max_index[0,-1])
            best_fit_max_cutoff = ((bin_edges[best_fit_max_index]) + (bin_edges[best_fit_max_index + 1])) /2
            
            best_fit_max_percentile_closest = best_fit_max * percentile_intensity_cutoff
            best_fit_data_percentile_cutoff = best_fit_line - best_fit_max_percentile_closest
            best_fit_data_percentile_cutoff_index = np.asarray((np.where(best_fit_data_percentile_cutoff < 0)))
            
            best_fit_data_percentile_cutoff_index = best_fit_data_percentile_cutoff_index[best_fit_data_percentile_cutoff_index >best_fit_max_index ]
            best_fit_data_percentile_cutoff_index = best_fit_data_percentile_cutoff_index[0]
            
            cutoff = bin_edges[best_fit_data_percentile_cutoff_index]
            
            if use_hist_max_cutoff == True:
                histogram = histogram[1::]
                bin_edges = bin_edges[1::]
                histmax = np.max(histogram)
                hist_max_index = int(np.asarray((np.where(histogram == histmax))))
                hist_max_cutoff = ((bin_edges[hist_max_index]) + (bin_edges[hist_max_index + 1])) /2
                cutoff = hist_max_cutoff
       
        binary_filtered = np.where(image_norm>cutoff,1,0) #cutoff is high
       
        
       
        
        distance = binary_filtered * image
      
        
        local_max_coords =feature.peak_local_max(distance, min_distance=1,num_peaks_per_label=1)#, threshold_abs=None, threshold_rel=None, exclude_border=True, indices=True, num_peaks=None, footprint=None, labels=None, num_peaks_per_label=None, p_norm=None)
        
        
        #%
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)
        
        segmented_cells = segmentation.watershed(-distance, markers, mask=binary_filtered)
        #%
        #Extract metadata based upon raw unprocessed image spots were created on
        props = ('label','mean_intensity','weighted_centroid','area')
        
        table1 = measure.regionprops_table(label_image=segmented_cells, intensity_image=image,properties=props)
        
        
        props = ('label','mean_intensity')    
        if i == 0:
            table2 = measure.regionprops_table(label_image=segmented_cells, intensity_image=image2,properties=props)
        if i == 1:
            table2 = measure.regionprops_table(label_image=segmented_cells, intensity_image=image1,properties=props)
             
        
        
        
        table1 = np.asarray(pd.DataFrame.from_dict(table1, orient='columns'))
        table2 = np.asarray(pd.DataFrame.from_dict(table2, orient='columns'))
        table1 = np.hstack((table1,table2[:,1].reshape(len(table2),1)))
        
        properties = measure.regionprops(label_image=segmented_cells, intensity_image=image)
        keep_spots = []
        current_coords = []
        current_coords_max = []
        current_coords_min = []
        current_coords_planes = []
        current_planes_min = []
        row = []
        
        
        for row in properties:
            if row.area < max_spot_size:
                if row.area > min_spot_size:
                    current_coords = row.coords
                    current_coords_min = np.min(current_coords, axis=0)
                    current_coords_max = np.max(current_coords, axis=0)
                    current_coords_planes = current_coords_max - current_coords_min
                    current_planes_min = np.min(current_coords_planes)
                    if current_planes_min >= 2:
                        keep_spots.append(row.label)
        
       ###ADD size and intensity filtering steps to segmented_cells data 
        
        if i == 0:
            #psd_spots = table1
            psd_spots = table1[np.isin(table1[:,0],keep_spots)]
            psd_spots_image =segmented_cells
            psd_keep = keep_spots
        if i == 1:
            #syn_spots = table1
            syn_spots = table1[np.isin(table1[:,0],keep_spots)]
            syn_spots_image = segmented_cells
            syn_keep = keep_spots
        
           
        
        
        #%
        
        del image_norm
        del distance
        del local_max_coords
        del current_coords
        del current_coords_max
        del current_coords_min
        del current_coords_planes
        del current_planes_min
        del markers
        del segmented_cells
        del properties
        del table1
        
    if display_detected_spots == True:
        
        viewer = napari.view_image(image1, rgb=False,scale=voxelsize, name='Reference channel',blending='additive',colormap='red')
        viewer.add_image(psd_spots_image , rgb=False,scale=voxelsize, name='Reference spot',blending='additive',colormap='green')
        viewer.add_image(image2, rgb=False,scale=voxelsize, name='Partner channel',blending='additive',colormap='blue')
        viewer.add_image(syn_spots_image , rgb=False,scale=voxelsize, name='Partner spot',blending='additive',colormap='yellow')
        if stop_after_display == True:
            all_data_variable = np.zeros((1,856))
            Normal_Freq = np.zeros((150,1))
            psd_meta_data = np.zeros((1,856))
            syn_meta_data = np.zeros((1,856))
            synaptic_loci = np.zeros((1,856))
            current_df = np.zeros((1,856))
            
            return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
        
       

    i2 = 1
    tic = time.perf_counter()

    if cross_filter_spots == False:
        S1sorteddata = np.flipud(psd_spots[psd_spots[:,1].argsort()])
        S2sorteddata = np.flipud(syn_spots[syn_spots[:,1].argsort()])
    elif cross_filter_spots == True and bottom_percent == False:
        S1sorteddata = np.flipud(psd_spots[psd_spots[:,6].argsort()])
        S2sorteddata = np.flipud(syn_spots[syn_spots[:,6].argsort()])
    elif cross_filter_spots == True and bottom_percent == True:
        S1sorteddata = (psd_spots[psd_spots[:,6].argsort()])
        S2sorteddata = (syn_spots[syn_spots[:,6].argsort()])    
    psd_spots = np.hstack((np.repeat('reference_spot',len(psd_spots)).reshape(len(psd_spots),1),psd_spots))
    syn_spots = np.hstack((np.repeat('partner_spots',len(syn_spots)).reshape(len(syn_spots),1),syn_spots))
    all_raw_spots_data = np.vstack((psd_spots,syn_spots))
    file_name3 =   re.split('/',file_name2)
    file_name3 = file_name3[len(file_name3)-1]
    all_raw_spots_data_df = pd.DataFrame(all_raw_spots_data)
    all_raw_spots_data_df.to_csv(path + "/" + out_name[0:4]    + "_" + file_name3 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff) + "_MT_" + str(mosaic_tile_i) + "_raw_spot_data.csv")               
   
    
    
    #Selects top X% spots
    S1numberspots = np.ma.size(S1sorteddata,axis=0)
    S1percentagespots = np.asarray((S1numberspots * percent)).astype(int)
    S1sorteddata2 = S1sorteddata[0:S1percentagespots,:]
       
    S1_nan_indices = np.where([np.logical_not(np.isnan(S1sorteddata2[:,3]))])
    S1_nan_indices = np.asarray(S1_nan_indices[1])
    S1sorteddata2 = S1sorteddata2[S1_nan_indices] 
       

    S1coords = S1sorteddata2[:,2:5]

    S1coords = np.asarray([S1coords[:,0]*voxelsize[0],S1coords[:,1]*voxelsize[1],S1coords[:,2]*voxelsize[2]]).T
      
    #Selects top X% spots
    S2numberspots = np.ma.size(S2sorteddata,axis=0)
    S2percentagespots = np.asarray((S2numberspots * percent)).astype(int)
    S2sorteddata2 = S2sorteddata[0:S2percentagespots,:]
       
    S2_nan_indices = np.where([np.logical_not(np.isnan(S2sorteddata2[:,3]))])
    S2_nan_indices = np.asarray(S2_nan_indices[1])
    S2sorteddata2 = S2sorteddata2[S2_nan_indices]       


    S2coords = S2sorteddata2[:,2:5]

    S2coords = np.asarray([S2coords[:,0]*voxelsize[0],S2coords[:,1]*voxelsize[1],S2coords[:,2]*voxelsize[2]]).T



    #%
    if S1coords.size > 0:
        if S2coords.size > 0:

            
            NNall=[]
            for row in S1coords:
                row = row.reshape(1,len(row))
                nn = scipy.spatial.distance.cdist(row, S2coords, metric='euclidean').T
                #nn = nn[np.logical_not(np.isnan(nn))]
                nn_indices = np.where([np.logical_not(np.isnan(nn))])
                nn_indices = np.asarray(nn_indices[1])
                nn = nn[nn_indices]
                S2sorteddata2 = S2sorteddata2[nn_indices,:]
                nn_min = np.min(nn)
                S2_partner = np.hstack([nn,S2sorteddata2])
                S2_partner_org =  S2_partner
                S2_partner = S2_partner[S2_partner[:,0]==nn_min]
                S2_partner = S2_partner[0,:]
                NNall.append(S2_partner)
        
            NNall = np.asarray(NNall)
            all_data = np.hstack([S1sorteddata2,NNall])
            if i2 == 0:
                all_data_fixed = pd.DataFrame(all_data)
            if i2 == 1:
                all_data_variable = pd.DataFrame(all_data)              
            edges = [0,	0.02,	0.04,	0.06,	0.08,	0.1,	0.12,	0.14,	0.16,	0.18,	0.2,	0.22,	0.24,	0.26,	0.28,	0.3,	0.32,	0.34,	0.36,	0.38,	0.4	,0.42	,0.44,	0.46	,0.48	,0.5,	0.52,	0.54	,0.56	,0.58,	0.6	,0.62	,0.64,	0.66,	0.68,	0.7,	0.72,	0.74	,0.76,	0.78,	0.8,	0.82,	0.84,	0.86	,0.88	,0.9,	0.92	,0.94,	0.96	,0.98,	1	,1.02,	1.04	,1.06,	1.08	,1.1,	1.12,	1.14,	1.16,	1.18,	1.2,	1.22	,1.24	,1.26	,1.28,	1.3	,1.32	,1.34,	1.36,	1.38	,1.4	,1.42	,1.44,	1.46	,1.48,	1.5,	1.52	,1.54	,1.56	,1.58,	1.6	,1.62	,1.64	,1.66,	1.68	,1.7,	1.72,	1.74	,1.76,	1.78,	1.8,	1.82,	1.84,	1.86,1.88	,1.9	,1.92,	1.94	,1.96,	1.98,	2	,2.02	,2.04	,2.06,	2.08,	2.1	,2.12,	2.14	,2.16,	2.18,	2.2,	2.22	,2.24,	2.26	,2.28,	2.3	,2.32,	2.34,	2.36	,2.38,	2.4,	2.42	,2.44,	2.46,	2.48,	2.5,	2.52	,2.54,	2.56	,2.58	,2.6,	2.62,	2.64	,2.66	,2.68,	2.7	,2.72,	2.74,	2.76,	2.78	,2.8,	2.82	,2.84,	2.86	,2.88,	2.9,	2.92,	2.94,	2.96,	2.98	,3];
            Normal_Freq,bins = np.histogram(NNall[:,0],bins=edges)
            Normal_Freq = Normal_Freq.reshape(150,1)
        
            psd_spots_image_raw = psd_spots_image
            syn_spots_image_raw = syn_spots_image
        
#%       
            synaptic_loci = all_data[all_data[:,7]<synaptic_cutoff]
            dilated_mask = np.zeros((np.shape(image3)[0],np.shape(image3)[1],np.shape(image3)[2])).astype(np.uint8)
        
            if len(synaptic_loci) > 0:
                
                psd_synaptic_image = np.isin(psd_spots_image,synaptic_loci[:,0]).astype(np.uint8)
                syn_synaptic_image = np.isin(syn_spots_image,synaptic_loci[:,8]).astype(np.uint8)
                
                psd_spots_image_raw = psd_spots_image.copy()
                syn_spots_image_raw = syn_spots_image.copy()
                
                psd_spots_image = psd_spots_image * psd_synaptic_image
                syn_spots_image = syn_spots_image * syn_synaptic_image
                if use_pretrained_classifier == True:
                                        
                    
                    props = ('label','area','mean_intensity','min_intensity',
                             'max_intensity','major_axis_length','minor_axis_length',
                             'inertia_tensor_eigvals')
                    spot1_self = measure.regionprops_table(label_image=psd_spots_image, intensity_image=image1,properties=props)
                    spot1_self = np.asarray(pd.DataFrame.from_dict(spot1_self, orient='columns'))
                    
                    spot2_self = measure.regionprops_table(label_image=syn_spots_image, intensity_image=image2,properties=props)
                    spot2_self = np.asarray(pd.DataFrame.from_dict(spot2_self, orient='columns'))
    
    
                       
                    #Extract metadata based upon raw unprocessed image of other channel
                    props = ('label','mean_intensity','min_intensity','max_intensity')
                    spot1_other = measure.regionprops_table(label_image=psd_spots_image, intensity_image=image2,properties=props) 
                    spot1_other = np.asarray(pd.DataFrame.from_dict(spot1_other, orient='columns'))
                    
                    
                    spot2_other = measure.regionprops_table(label_image=syn_spots_image, intensity_image=image1,properties=props) 
                    spot2_other = np.asarray(pd.DataFrame.from_dict(spot2_other, orient='columns'))
              
                    spot1_properties = measure.regionprops(label_image=psd_spots_image, intensity_image=image1)# 
                    spot2_properties = measure.regionprops(label_image=syn_spots_image, intensity_image=image1)# 
                                 
                      
                    
    
    
                    
                    bases = [2,10,np.e]
                    spot1_entropy_all = []
                    spot2_entropy_all = []
                    spot1_skew_other_all = []
                    spot2_skew_other_all = []                            
    
                    for row in spot1_properties:
      
                        intensity_image1 = row.image_intensity
                        intensity_image2 = np.where(intensity_image1 > 0 , intensity_image1 , np.nan)
                        label = row.label
    
                        entropy_current = []
                        for base_value in bases:
                            entropy = measure.shannon_entropy(intensity_image2, base=base_value)
                            entropy_current = np.hstack([entropy_current,entropy])
                        spot1_entropy_all.append(entropy_current)
                            
                        #intensity_image = row.image_intensity
                        intensity_image_ravel = np.ravel(intensity_image1)
                        intensity_image_ravel = intensity_image_ravel[np.nonzero(intensity_image_ravel)]
                        skew_current = scipy.stats.skew(intensity_image_ravel)
                        kurtosis_current = scipy.stats.kurtosis(intensity_image_ravel)
                        std_current = np.std(intensity_image_ravel)#.reshape(1,1)
                        mean_current = np.mean(intensity_image_ravel)
                        cv_current =  std_current /  mean_current * 100
                        
                        median_current = np.median(intensity_image_ravel)
                        #std_median_current = std_current / median_current
                        
                        
                        skew_other = np.hstack([label,skew_current,kurtosis_current, std_current,cv_current,median_current])
                        spot1_skew_other_all.append(skew_other) 
     
    
                    spot1_entropy_all = np.asarray(spot1_entropy_all)
                    spot1_skew_other_all = np.asarray(spot1_skew_other_all) 
    
                    for row in spot2_properties:
      
                        intensity_image1 = row.image_intensity
                        intensity_image2 = np.where(intensity_image1 > 0 , intensity_image1 , np.nan)
                        label = row.label
    
                        entropy_current = []
                        for base_value in bases:
                            entropy = measure.shannon_entropy(intensity_image2, base=base_value)
                            entropy_current = np.hstack([entropy_current,entropy])
                        spot2_entropy_all.append(entropy_current)
                            
                        #intensity_image = row.image_intensity
                        intensity_image_ravel = np.ravel(intensity_image1)
                        intensity_image_ravel = intensity_image_ravel[np.nonzero(intensity_image_ravel)]
                        skew_current = scipy.stats.skew(intensity_image_ravel)
                        kurtosis_current = scipy.stats.kurtosis(intensity_image_ravel)
                        std_current = np.std(intensity_image_ravel)#.reshape(1,1)
                        mean_current = np.mean(intensity_image_ravel)
                        cv_current =  std_current /  mean_current * 100
                        
                        median_current = np.median(intensity_image_ravel)
                        
                        
                        
                        skew_other = np.hstack([label,skew_current, kurtosis_current,std_current,cv_current,median_current])
                        spot2_skew_other_all.append(skew_other) 
     
    
                    
                    spot2_entropy_all = np.asarray(spot2_entropy_all)
                    spot2_skew_other_all = np.asarray(spot2_skew_other_all) 
    
    
                    psd_properties = measure.regionprops(label_image=psd_spots_image)# 
                    syn_properties = measure.regionprops(label_image=syn_spots_image)# 
                    psd_label_ids = []
                    syn_label_ids = []
                    for row in psd_properties:
                        row_label = row.label
                        psd_label_ids.append(row_label)
                    for row in syn_properties:
                        row_label = row.label
                        syn_label_ids.append(row_label)
    
    
                    psd_label_ids = np.vstack(psd_label_ids)
                    syn_label_ids = np.vstack(syn_label_ids)
                    all_rows_coords = []
                    for row in synaptic_loci:
                        current_psd_id = row[0]
                        current_syn_id = row[8]
                        psd_prop_row = int(np.where(psd_label_ids == current_psd_id)[0])
                        syn_prop_row = int(np.where(syn_label_ids == current_syn_id)[0])
                        psd_coords = psd_properties[psd_prop_row].coords
                        syn_coords = syn_properties[syn_prop_row].coords
                        matched_row = []
                        for coords_row in psd_coords:
                            matched0 = np.isin(syn_coords[:,0],coords_row[0])
                            matched1 = np.isin(syn_coords[:,1],coords_row[1])
                            matched2 = np.isin(syn_coords[:,2],coords_row[2])
                            matched = np.hstack((matched0.reshape(len(matched0),1),matched1.reshape(len(matched0),1),matched2.reshape(len(matched0),1)))
                            
                            matched_sum = np.sum(matched,axis=1)
                            if np.max(matched_sum) == 3:
                                matched_row.append(1)
                            if np.max(matched_sum) != 3:
                                matched_row.append(0)
                        total_psd = len(psd_coords)
                        total_syn = len(syn_coords)
                        total_matched = np.sum(matched_row)
                        percent_psd = total_matched / total_psd * 100
                        percent_syn = total_matched / total_syn * 100
                        all_coords_data = np.hstack((current_psd_id,current_syn_id,total_psd, total_syn, total_matched, percent_psd, percent_syn))
                        all_rows_coords.append( all_coords_data)
                    all_rows_coords = np.vstack(all_rows_coords)
                    
                    #%
                    
                    spot1_entropy_skew = np.hstack((spot1_skew_other_all[:,0].reshape(np.shape(spot1_skew_other_all)[0],1),
                                                    spot1_entropy_all,
                                                    spot1_skew_other_all[:,1::]))
                    spot2_entropy_skew = np.hstack((spot2_skew_other_all[:,0].reshape(np.shape(spot2_skew_other_all)[0],1),
                                                    spot2_entropy_all,
                                                    spot2_skew_other_all[:,1::]))                
                    
                    #% reorder metadata metadata
                    spot1_self_df = pd.DataFrame(spot1_self,index=spot1_self[:,0])
                    spot1_self_df = spot1_self_df.reindex(labels=spot1_self_df[0],index=synaptic_loci[:,0])
                            
                    spot2_self_df = pd.DataFrame(spot2_self,index=spot2_self[:,0])
                    spot2_self_df = spot2_self_df.reindex(labels=spot2_self_df[0],index=synaptic_loci[:,8])
                    
                    spot1_other_df = pd.DataFrame(spot1_other,index=spot1_other[:,0])
                    spot1_other_df = spot1_other_df.reindex(labels=spot1_other_df[0],index=synaptic_loci[:,0])
                            
                    spot2_other_df = pd.DataFrame(spot2_other,index=spot2_other[:,0])
                    spot2_other_df = spot2_other_df.reindex(labels=spot2_other_df[0],index=synaptic_loci[:,8])
                    
                    spot1_entropy_skew_df = pd.DataFrame(spot1_entropy_skew,index=spot1_entropy_skew[:,0])
                    spot1_entropy_skew_df = spot1_entropy_skew_df.reindex(labels=spot1_entropy_skew_df[0],index=synaptic_loci[:,0])
                            
                    spot2_entropy_skew_df = pd.DataFrame(spot2_entropy_skew,index=spot2_entropy_skew[:,0])
                    spot2_entropy_skew_df = spot2_entropy_skew_df.reindex(labels=spot2_entropy_skew_df[0],index=synaptic_loci[:,8])
                 
                    spot1_self_sorted = np.asarray(spot1_self_df)
                    spot2_self_sorted = np.asarray(spot2_self_df)
                    
                    
                    spot1_entropy_skew_sorted = np.asarray(spot1_entropy_skew_df)
                    spot2_entropy_skew_sorted = np.asarray(spot2_entropy_skew_df)
                    
                    
                    spot1_other_sorted = np.asarray(spot1_other_df)
                    spot2_other_sorted = np.asarray(spot2_other_df)
                    
                    spot1_test1 = np.sum(np.abs((synaptic_loci[:,0] - spot1_self_sorted[:,0])))
                    spot1_test2 = np.sum(np.abs((synaptic_loci[:,0] - spot1_other_sorted[:,0])))
                    spot1_test3 = np.sum(np.abs((synaptic_loci[:,0] - spot1_entropy_skew_sorted[:,0])))
                    
                    spot2_test1 = np.sum(np.abs((synaptic_loci[:,8] - spot2_self_sorted[:,0])))
                    spot2_test2 = np.sum(np.abs((synaptic_loci[:,8] - spot2_other_sorted[:,0])))
                    spot2_test3 = np.sum(np.abs((synaptic_loci[:,8] - spot2_entropy_skew_sorted[:,0])))
                    
                    overlap_test1 = np.sum(np.abs((synaptic_loci[:,0] - all_rows_coords[:,0])))
                    overlap_test2 = np.sum(np.abs((synaptic_loci[:,8] - all_rows_coords[:,1])))
                    
                    #%
                    if (spot1_test1 == 0) and (spot1_test2 == 0) and (spot1_test3 == 0) and (spot2_test1 == 0) and (spot2_test2 == 0) and (spot2_test3 == 0) and (overlap_test1 == 0) and (overlap_test2 == 0):
                        print('all true')
                        classifier_data = np.hstack((spot1_self_sorted[:,1::],spot1_other_sorted[:,1::],spot1_entropy_skew_sorted[:,1::],
                                                    spot2_self_sorted[:,1::],spot2_other_sorted[:,1::],spot2_entropy_skew_sorted[:,1::],
                                                    all_rows_coords[:,5::]))
                        y_val = model_reload.predict(classifier_data) 
                        keep_rows = np.where(y_val=='synaptic',True,False)
                        synaptic_loci = synaptic_loci[keep_rows,:]
                        synaptic_loci = all_data[all_data[:,7]<synaptic_cutoff_classifier]
                        psd_synaptic_image = np.isin(psd_spots_image,synaptic_loci[:,0]).astype(np.uint8)
                        syn_synaptic_image = np.isin(syn_spots_image,synaptic_loci[:,8]).astype(np.uint8)
                        psd_spots_image = psd_spots_image * psd_synaptic_image
                        syn_spots_image = syn_spots_image * syn_synaptic_image      
                    else:
                        print('labels dont match. Not filtering data.')
                
                
                #%
                
                
                if np.shape(synaptic_loci)[0] > 0:
                    if (label_loci_to_mask == True) and (running_channel_alignment == False):    
    
                        ellipse_radius = (int(np.round(nm_cutoff_neurite / voxelsize[1])))
                        ellipse_array = np.zeros(((ellipse_radius*2)+1,(ellipse_radius*2)+1,(ellipse_radius*2)+1))
                        center = [ellipse_radius,ellipse_radius,ellipse_radius]
                        #%
                        #radius = ellipse_radius
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
                        
                        image_dims= np.shape(image3)
                        ## ADS added 5/20/2024 to stop failed concat below on empty array
                        all_neurite_distances = []#['fail_detected']
                        all_neurite_radii  = []#['fail_detected']
                        synaptic_centroids = np.asarray([(synaptic_loci[:,2]+synaptic_loci[:,10])/2,
                                                         (synaptic_loci[:,3]+synaptic_loci[:,11])/2,
                                                         (synaptic_loci[:,4]+synaptic_loci[:,12])/2]).T
                      
                        
                        
        
                        
                        
                        scaling_factor = voxelsize[0] / voxelsize[1]
                        
                        
                        low_gauss = filters.gaussian(image3,sigma=[1,np.round(scaling_factor),np.round(scaling_factor)])
                        high_gauss = filters.gaussian(image3,sigma=[2,2*np.round(scaling_factor),2*np.round(scaling_factor)])
                        filtered = (low_gauss - high_gauss)
                        
                        
                        filtered_cutoff = np.where(filtered > neurite_mask_gauss_quantile_cutoff ,1,0)
                        
                       
                        
                        ## Merge mask with raw intensity
                        is_this_real = filtered_cutoff * image3
                        
                        ##Select mask above cutoff
                        neurite_mask = np.where(is_this_real > neurite_mask_intensity_cutoff,255,0)
                        
                        #neurite_mask = np.max(above_neurite_cutoff,axis=0)
                        
                        #%
                        above_neurite_cutoff = np.where(is_this_real > neurite_mask_intensity_cutoff,1,0)
                        above_neurite_cutoff_bool = above_neurite_cutoff.astype(bool)
                        dilated_mask = morphology.binary_dilation(above_neurite_cutoff_bool,footprint=ellipse_array)
                        dilated_mask = dilated_mask.astype(np.uint8)
                        
                        centroids = np.round(synaptic_centroids).astype(np.uint16)
                        
                        
                        neurite_positive = dilated_mask[centroids[:,0],centroids[:,1],centroids[:,2]]
                        
                         
                        grid_sizes = [1748,874,437]
                        
                        current_image_subfield_data = np.asarray(['grid_size', 'current_sub_grid', 'x_start', 'y_start', 'x_end', 'y_end', 'total_field_area',
                                                                  'total_neurite_area','non_neurite_intensity_mean','neurite_intensity_mean','non_neurite_intensity_median','neurite_intensity_median']).reshape(1,12)
                        for grid_size in grid_sizes:
                           
                            current_sub_grid = 0
                            x_steps = int(np.shape(dilated_mask)[1] / grid_size)
                            y_steps = int(np.shape(dilated_mask)[2] / grid_size)
                            
                            for current_x in range(0,x_steps):
                                
                                for current_y in range(0,y_steps):
                                    
                                    x_start = current_x * grid_size
                                    y_start = current_y * grid_size
                                    x_end = x_start + grid_size
                                    y_end = y_start + grid_size
                                    current_sub_field = dilated_mask[:,y_start:y_end,x_start:x_end]
                                    current_sub_field_intensity = image3[:,y_start:y_end,x_start:x_end]
                                    total_neurite_area = np.sum(current_sub_field)
                                    total_field_area = np.shape(current_sub_field)[0] * np.shape(current_sub_field)[1] * np.shape(current_sub_field)[1]
                                    
                                    non_neurite_intensity = current_sub_field_intensity[dilated_mask[:,y_start:y_end,x_start:x_end]==0]
                                    neurite_intensity = current_sub_field_intensity[dilated_mask[:,y_start:y_end,x_start:x_end]==1]
                                  
                                    
                                    current_subfield = np.hstack((grid_size,current_sub_grid,x_start,y_start,x_end,y_end,total_field_area,total_neurite_area,
                                                                  np.mean(non_neurite_intensity),np.mean(neurite_intensity),np.median(non_neurite_intensity),np.median(neurite_intensity)))
                                    current_image_subfield_data = np.vstack((current_image_subfield_data,current_subfield))
                                    current_sub_grid = current_sub_grid + 1
                        
                        #%
                        
                        current_image_subfield_data = pd.DataFrame(current_image_subfield_data)
                        
                      
                        
                       
                        for row in synaptic_centroids:
        
                            current_synaptic_loci_centroid_int_neurite = np.round(row).astype(np.int16).reshape(1,3)
                            ##Create binarized mask of 3rd channel
                            
                            xy_radius_scale = 1
                            
                            radius = neurite_field_voxel_radius                                                            
                            
                            radius_check = neurite_field_voxel_radius 
                            if max_neurite_distance_variable == True:
                                min_distances = row * voxelsize
                                max_distances = np.asarray([(image_dims[0]-1) - row[0],(image_dims[1]-1) - row[1],(image_dims[2]-1) - row[2]])* voxelsize
                                min_radius = np.min(np.hstack((min_distances,max_distances,max_radius)))
                                max_neurite_distance = min_radius
                            
                            
                            
                            
                            
                            # Create integers for coordinate adjustement to set maximum possible spot size
                            coords = np.linspace(-(radius * xy_radius_scale),(radius * xy_radius_scale),num=(((radius * xy_radius_scale)*2)+1),dtype=int)
                            coords = coords.astype(np.int16)
                            pairs = np.asarray(np.meshgrid(coords,coords,coords)).T.reshape(-1,3)
                            pairs_nm = pairs * voxelsize
                            X= np.sqrt((pairs[:,0]**2)+(pairs[:,1]**2)+(pairs[:,2]**2))
                            X_nm= np.sqrt((pairs_nm[:,0]**2)+(pairs_nm[:,1]**2)+(pairs_nm[:,2]**2))
                            
                            
                            
                            
                            keep_nm_neurite = pairs[X_nm<=max_neurite_distance]
        
                            
                            current_synaptic_field_neurite = current_synaptic_loci_centroid_int_neurite + keep_nm_neurite
                            
                            
                            current_synaptic_field_neurite_org = current_synaptic_field_neurite.copy()
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,0] >=0 ]
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,0] <=(image_dims[0]-1) ]
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,1] >=0 ]
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,1] <=(image_dims[1]-1) ]
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,2] >=0 ]
                            current_synaptic_field_neurite = current_synaptic_field_neurite[current_synaptic_field_neurite[:,2] <=(image_dims[2]-1) ]
                            number_synaptic_voxels = len(current_synaptic_field_neurite)
                            
                            if number_synaptic_voxels == np.shape(keep_nm_neurite)[0]:
                                neurite_subfield_intensities = neurite_mask[current_synaptic_field_neurite[:,0],
                                                                            current_synaptic_field_neurite[:,1],
                                                                            current_synaptic_field_neurite[:,2]]
                                
                                neurite_subfield_intensities = neurite_subfield_intensities.reshape(number_synaptic_voxels,1)
                                current_synaptic_field_neurite = np.hstack((current_synaptic_field_neurite,
                                                                            neurite_subfield_intensities,
                                                                            np.zeros((number_synaptic_voxels,1)))) 
                                ## 1/29/2023 ADS change '==1' to '==255' to align with new DOG method of neurite masking
                                rows_with_neurites = current_synaptic_field_neurite[current_synaptic_field_neurite[:,3]==255,0:3]
                               
                                if np.size(rows_with_neurites) == 0:
                                    closest_neurite = max_neurite_distance + .001#['greater than_' +str(max_neurite_distance)]
                                elif np.size(rows_with_neurites) > 0:
                                    rows_with_neurites = np.asarray([rows_with_neurites[:,0]*voxelsize[0],rows_with_neurites[:,1]*voxelsize[1],rows_with_neurites[:,2]*voxelsize[2]]).T    
                                    current_synaptic_loci_centroid_int_neurite = current_synaptic_loci_centroid_int_neurite*voxelsize
                                    nn = scipy.spatial.distance.cdist(current_synaptic_loci_centroid_int_neurite, rows_with_neurites, metric='euclidean').T
                                    closest_neurite = np.min(nn)
                            elif number_synaptic_voxels != np.shape(keep_nm_neurite)[0]:
                                closest_neurite = 'not full field'
                            all_neurite_distances.append(closest_neurite)
                            all_neurite_radii.append(max_neurite_distance)
                        all_neurite_distances = np.vstack(all_neurite_distances)
                        all_neurite_radii = np.vstack(all_neurite_radii)
                        
                        ## ADS added 5/20/2024 to stop failed concat above on empty array
                        #all_neurite_distances = all_neurite_distances[1::,0]
                        #all_neurite_radii = all_neurite_radii[1::,0]
                     
                        neurite_positive = neurite_positive.reshape(len(neurite_positive),1)
                        neurite_positive_labels = np.asarray(['label','mean','z','y','x','area','other intensity','nn','label','mean','z','y','x','area','other intensity','all_neurite_distances','all_neurite_radii','neurite_positive'])
                        neurite_positive_labels = neurite_positive_labels.reshape(1,len(neurite_positive_labels))
                        neurite_positive = np.vstack((neurite_positive_labels,np.hstack((synaptic_loci,all_neurite_distances,all_neurite_radii,neurite_positive))))
                        
                        neurite_positive = pd.DataFrame(neurite_positive)
                                                                                                
            
            
                    
                    if (label_loci_to_mask == True) and (running_channel_alignment == False):
                        synaptic_loci = np.hstack((synaptic_loci,all_neurite_distances,all_neurite_radii))
                    del all_data
                    del nn
                    del nn_indices
                    del nn_min
                    del NNall
                    del psd_keep
                    del psd_spots
                    del psd_spots_image_raw
                    del psd_synaptic_image
                    del syn_keep
                    del syn_spots
                    del syn_spots_image_raw
                    del syn_synaptic_image
                    del S1_nan_indices
                    del S1coords
                    del S1sorteddata
                    del S1sorteddata2
                    del S2_nan_indices
                    del S2coords
                    del S2sorteddata
                    del S2sorteddata2
                    del S2_partner_org
                   
                    for i in range(0,2):
                        
                      
                        if i == 0:
                            
                            
                            
                            tic = time.perf_counter()
                            psd_meta_data, all_3rd_channel_intensity_data, all_3rd_channel_intensity_data_flipped,all_rows_coords = get_metadata(psd_spots_image,image1, image2,image3,synaptic_loci,i,psd_spots_image, syn_spots_image,dilated_mask)
                            toc = time.perf_counter()
                            print("process time raw variable spots metadata psd: ")
                            print(f"time {toc - tic:0.4f} seconds")#% 
                            
                        if i == 1:
                            tic = time.perf_counter()
                            syn_meta_data = get_metadata(syn_spots_image,image2, image1,image3,synaptic_loci,i,psd_spots_image, syn_spots_image,dilated_mask)
                            toc = time.perf_counter()
                            print("process time raw variable spots metadata syn: ")
                            print(f"time {toc - tic:0.4f} seconds")#% 
        
             
                    file_name2 =   re.split('/',file_name2)
                    file_name2 = file_name2[len(file_name2)-1]     
                    np.savetxt(path + "/" + out_name[0:4]   + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) + "_synaptic_NN_data.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
                    np.savetxt(path + "/" + out_name[0:4]   + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) +  "_Normal_Freq.csv",Normal_Freq,delimiter=",") #+ str(number_dropped)
                   
                    psd_df = pd.DataFrame(psd_meta_data,index=psd_meta_data[:,0])
                    psd_df = psd_df.reindex(labels=psd_df[0],index=synaptic_loci[:,0])
                
                    #% reorder synapsin metadata
                
                    syn_df = pd.DataFrame(syn_meta_data,index=syn_meta_data[:,0])
                    syn_df = syn_df.reindex(labels=syn_df[0],index=synaptic_loci[:,8])
                    
                    all_coords_df = pd.DataFrame(all_rows_coords,index=all_rows_coords[:,0])
                    all_coords_df = all_coords_df.reindex(labels=all_coords_df[0],index=synaptic_loci[:,0])
                    
                    #%
                    filename_rep = np.repeat(file_name, psd_df.shape[0]).reshape(psd_df.shape[0],1)
                    current_df = np.hstack((filename_rep,synaptic_loci[:,7].reshape(len(synaptic_loci),1),psd_df,syn_df,all_coords_df))
                    current_df = pd.DataFrame(current_df)
                    current_df.to_csv(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) +  "_PSD_SYN_metadata.csv")
                    if (process_variable_radii == True) and (running_channel_alignment == False): 
                        current_df2 = np.hstack((all_3rd_channel_intensity_data, all_3rd_channel_intensity_data_flipped))
                        current_df2 = pd.DataFrame(current_df2)
                        current_df2.to_csv(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) +  "_3rd_channel_vRadii.csv")               
                        
                    if (label_loci_to_mask == True) and (running_channel_alignment == False):   
                        current_image_subfield_data.to_csv(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) + '_subfield_neurite_data.csv')
                        neurite_positive.to_csv(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number) + "_BLS_" + str(third_channel_bg_cutoff)  + "_MT_" + str(mosaic_tile_i) + '_neurite_positive_data.csv')
                        
                    
                    
                    del S1numberspots
                    del S1percentagespots
                    del S2_partner
                    del S2numberspots
                    del S2percentagespots
                    del all_3rd_channel_intensity_data
                    del all_3rd_channel_intensity_data_flipped
                    del all_coords_df
                    del all_raw_spots_data
                    del all_raw_spots_data_df
                    del all_rows_coords
                    del binary_filtered
                    del bins
                    del cutoff
                    del edges
                    del file_name
                    del file_name2
                    del file_name3
                    del filename_rep
                    del i
                    del i2
                    del i4
                    del image
                    del image1
                    del image2
                    del image3
                    del keep_spots
                    del local_max_mask
                    del maxi
                    del out_name
                    del props
                    del psd_df
                    del psd_spots_image
                    del qi
                    del row
                    del syn_df
                    del syn_spots_image
                    del table2
                    del third_channel_bg_cutoff
                    del tic
                    del tile_number
                    del toc
                    gc.collect()
                                   
                    save_df_variables = pd.DataFrame(dir())
                    save_df_variables.to_csv(path + '_' + '_spotsDetection_variables.txt',header=False, index=False)
                   
                    return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
                    del image1
                    del image2
                    del image3
                else:
                    file_name2 =   re.split('/',file_name2)
                    file_name2 = file_name2[len(file_name2)-1]     
                    all_data_variable = np.zeros((1,856))
                    Normal_Freq = np.zeros((150,1))
                    psd_meta_data = np.zeros((1,856))
                    syn_meta_data = np.zeros((1,856))
                    synaptic_loci = np.zeros((1,856))
                    current_df = np.zeros((1,856))
                    file_name2 =   re.split('/',file_name2)
                    file_name2 = file_name2[len(file_name2)-1]     
                    np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 +"_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_synaptic_NN_data_postClass.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
                    np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 +"_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_Normal_Freq_postClass.csv",Normal_Freq,delimiter=",") #+ str(number_dropped)
                    return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df                       
            else:
                file_name2 =   re.split('/',file_name2)
                file_name2 = file_name2[len(file_name2)-1]     
                all_data_variable = np.zeros((1,856))
                Normal_Freq = np.zeros((150,1))
                psd_meta_data = np.zeros((1,856))
                syn_meta_data = np.zeros((1,856))
                synaptic_loci = np.zeros((1,856))
                current_df = np.zeros((1,856))
                file_name2 =   re.split('/',file_name2)
                file_name2 = file_name2[len(file_name2)-1]     
                np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 +"_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_synaptic_NN_data.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
                np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 +"_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_Normal_Freq.csv",Normal_Freq,delimiter=",") #+ str(number_dropped)
                return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df                
                
            
        else:
            file_name2 =   re.split('/',file_name2)
            file_name2 = file_name2[len(file_name2)-1]     
            all_data_variable = np.zeros((1,856))
            Normal_Freq = np.zeros((150,1))
            psd_meta_data = np.zeros((1,856))
            syn_meta_data = np.zeros((1,856))
            synaptic_loci = np.zeros((1,856))
            current_df = np.zeros((1,856))
            file_name2 =   re.split('/',file_name2)
            file_name2 = file_name2[len(file_name2)-1]     
            np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_synaptic_NN_data.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
            np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_Normal_Freq.csv",Normal_Freq,delimiter=",") #+ str(number_dropped)
            return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
    
    else:
        file_name2 =   re.split('/',file_name2)
        file_name2 = file_name2[len(file_name2)-1]     
        all_data_variable = np.zeros((1,856))
        Normal_Freq = np.zeros((150,1))
        psd_meta_data = np.zeros((1,856))
        syn_meta_data = np.zeros((1,856))
        synaptic_loci = np.zeros((1,856))
        current_df = np.zeros((1,856))
        np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_synaptic_NN_data.csv",all_data_variable,delimiter=",") #+ str(number_dropped) + out_name removed 7
        np.savetxt(path + "/" + out_name[0:4]    + "_" + file_name2 + "_tile_" + str(tile_number)  + "_MT_" + str(mosaic_tile_i) + "_NoSPOTS_Normal_Freq.csv",Normal_Freq,delimiter=",") #+ str(number_dropped)
        return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
   
#@delayed
#@wrap_non_picklable_objects
def GetCZImetadata(file_current_test):
    current_metadatadict_czi =(czi.CziFile(file_current_test)).metadata(raw=False)
    
    #Find current image tile name
    curent_image_tile_ID = re.split('-',file_current_test)
    curent_image_tile_ID = curent_image_tile_ID[-1]
    curent_image_tile_ID = curent_image_tile_ID[0:-4]

    current_image_tile_number = int((np.where(tile_region_names==curent_image_tile_ID))[0])

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
              [current_image_tile_number]
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
              [current_image_tile_number]
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
    
    save_df_variables = pd.DataFrame(dir())
    save_df_variables.to_csv(path + '_' + '_scziMetadata.txt',header=False, index=False)

    return current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY


def spot_detection_for_channel_algnment(current_row_file,current_channel_to_align,reference_channel_chA,mosaic_tile_i):
    
    file_current_test = str(current_row_file)
    img = AICSImage(file_current_test,reconstruct_mosaic=False) 
    lazy_t0 = img.get_image_dask_data("CZYX", M=mosaic_tile_i)  
    image = lazy_t0.compute()
    if (chA_calculated == False) or (tile_scan_980 == False):
        tile_number = 0
        print('running this block')
        #current_tile = image
        image1, image2, image3,third_channel_bg_cutoff = select_channels(image,shift_raw_ch0,shift_raw_ch1,shift_raw_ch2,spot_1_channel,spot_2_channel,spot_3_channel,current_channel_to_align,reference_channel_chA)
        #return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
        image1filtered = np.where(image1>wmbls1,image1-wmbls1,0)
        image2filtered = np.where(image2>wmbls2,image2-wmbls2,0)
        
        
        for i in range (1,2):
            if i==1:
                qi = []
                image_name = image1filtered
                image_name_raw = image1
                image_other_raw = image2
                file_name = filename_slicing(file_current_test) #re.search(name_start, file)
                name = file_name#.group(1)
                #nameint = int(name[1:4])
                file_name = (name + 'channel_' + str(i))#(file_name.group(1) + 'channel_' + str(i))
                

                                                                                                                     
                psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df = spots_detection(image_name_raw,image_other_raw,file_name,out_name,path,i,qi,file_name,image3,tile_number,mosaic_tile_i,third_channel_bg_cutoff)
                return psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df
    if (chA_calculated == True) and (tile_scan_980 == True):
        current_tile_columns, current_tile_rows, current_tile_SizeX, current_tile_SizeY = GetCZImetadata(file_current_test)
        
       
        
        ####Manual disable for manual override
        actual_X_image_pixels = current_tile_SizeX - (24 * (current_tile_columns-1))
        actual_Y_image_pixels = current_tile_SizeY - (24 * (current_tile_rows-1))
        ## Manual override. If kept move to beginning     
        actual_X_tile_size = int(actual_X_image_pixels / current_tile_columns) #int(1736) if need manual override
        actual_Y_tile_size = int(actual_Y_image_pixels / current_tile_rows) #int(1736)
    
        slice_coordinates = np.zeros(((current_tile_rows * current_tile_columns), 2))
        tile_for_coordinates = 0

        for row_position in range(0,current_tile_columns):
            for column_position in range(0,current_tile_rows):
                slice_coordinates[tile_for_coordinates,0] = ((actual_Y_tile_size * column_position) + (24 * column_position))
                slice_coordinates[tile_for_coordinates,1] = ((actual_X_tile_size * row_position) + (24 * row_position))
                tile_for_coordinates = tile_for_coordinates + 1
                
        slice_coordinates = slice_coordinates.astype(np.uint16) 
        tile_number = 0


        for slice_coordinates_i in range(0,len(slice_coordinates)):
            slice_coordinates_row = slice_coordinates[slice_coordinates_i,:]
            current_tile = image[:,:,:,:,
                             slice_coordinates_row[0]:slice_coordinates_row[0]+actual_Y_tile_size,
                             slice_coordinates_row[1]:slice_coordinates_row[1]+actual_X_tile_size,
                             :]       
            image1, image2, image3,third_channel_bg_cutoff = select_channels(current_tile,shift_raw_ch0,shift_raw_ch1,shift_raw_ch2,spot_1_channel,spot_2_channel,spot_3_channel,current_channel_to_align,reference_channel_chA)
            
            
            
            
            #Create new baseline subtratction images
            image1filtered = np.where(image1>wmbls1,image1-wmbls1,0)
            image2filtered = np.where(image2>wmbls2,image2-wmbls2,0)
            
            #Perform channel 1 and then channel 2 spots detection 
            for i in range (1,2):
                if i==1:
                    qi = []
                    image_name = image1filtered
                    image_name_raw = image1
                    image_other_raw = image2
                    file_name = filename_slicing(file_current_test)
                    name = file_name#.group(1)
                    file_name = (name + 'channel_' + str(i))#(file_name.group(1) + 'channel_' + str(i))
                    psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df = spots_detection(image_name_raw,image_other_raw,file_name,out_name,path,i,qi,file_name,image3,tile_number,mosaic_tile_i)
                    
                elif i==2:
                    qi = []
                    image_name = image2filtered
                    image_name_raw = image2
                    image_other_raw =image1
                    file_name = file_current_test[0:-4] #re.search(name_start, file)
                    name = file_name
                    file_name = (name + 'channel_' + str(i))#(file_name.group(1) + 'channel_' + str(i))
                    spots_detection(image_name_raw,image_other_raw,file_name,out_name,path,i,qi)
            tile_number = tile_number + 1 
            #
print('running')

if (channel_align_image == True) and (channel_alignment_override == False):
    running_channel_alignment = True
    percent = percent_chA
    chA_calculated = False
    chA_image = True
    all_cha_conversion = ['image','mosaic_tile','reference_ch','aligned_ch','Z-median','Y-median','X-median','Z-std','Y-std','X-std']
    ch0_alignment= np.asarray([0,0,0]).reshape(1,3)
    ch1_alignment= np.asarray([0,0,0]).reshape(1,3)
    ch2_alignment= np.asarray([0,0,0]).reshape(1,3)
    for chA_file in chA_files:
        key_cha_img = AICSImage(chA_file,reconstruct_mosaic=False)
        number_chA_tiles = np.shape(key_cha_img)[0]
       
        for mosaic_tile_i in range(0,number_chA_tiles):
            for current_channel_to_align in channels_to_align:
            
                psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df = spot_detection_for_channel_algnment(chA_file,current_channel_to_align,reference_channel_chA,mosaic_tile_i)
            #%  
                current_df = current_df[current_df.iloc[:,1].astype(float)<.60]
                current_df =  np.asarray(current_df)
                
                raw_data =  np.flipud(current_df[current_df[:,4].argsort()])
                
                
                #%
                psd_spots =  current_df[:,5:8]
                syn_spots =  current_df[:,438:441] #ORG dded 6 to match synaptic field meta data
                S1coords = psd_spots.astype(float)
                S2coords = syn_spots.astype(float)
                all_partners = []
                all_nn = []
                for row in S1coords:
                    row = row.reshape(1,len(row))
                    nn = scipy.spatial.distance.cdist(row, S2coords, metric='euclidean').T
                    nn_min = np.min(nn)
                    partner_coords = S2coords[nn[:,0]==nn_min]
                    partner_coords = partner_coords[0,:].reshape(1,3)
                    all_partners.append(partner_coords)
                    all_nn.append(nn_min)
                all_nn = np.vstack(all_nn)
                all_partners = np.vstack(all_partners)
                
                xyz_conversion = S1coords - all_partners ##fixes Z needing negative
                
                    
                current_cha_conversion = ((np.median(xyz_conversion,axis=0).reshape(1,3)))
                current_cha_conversion_std = ((np.std(xyz_conversion,axis=0).reshape(1,3)))
                current_cha_conversion = np.hstack((np.asarray(chA_file).reshape(1,1),np.asarray(mosaic_tile_i).reshape(1,1),np.asarray(reference_channel_chA).reshape(1,1),
                                                    np.asarray(current_channel_to_align).reshape(1,1),current_cha_conversion,current_cha_conversion_std))
                all_cha_conversion = np.vstack((all_cha_conversion,current_cha_conversion))
                print(all_cha_conversion)
                #alignment_by_image = np.vstack((alignment_by_image,current_cha_conversion))
            
                if (current_channel_to_align == 0):
                    ch0_alignment = np.vstack((ch0_alignment,xyz_conversion))
                    
                if (current_channel_to_align == 1):
                    ch1_alignment = np.vstack((ch1_alignment,xyz_conversion))  
                    
                if (current_channel_to_align == 2):
                    ch2_alignment = np.vstack((ch2_alignment,xyz_conversion))
                    
        
    if len(ch0_alignment) > 1:
        ch0_alignment = ch0_alignment[1::,:]
       
    if len(ch1_alignment) > 1:
        ch1_alignment = ch1_alignment[1::,:]
        
    if len(ch2_alignment) > 1:
        ch2_alignment = ch2_alignment[1::,:]
        
        
        
        
        
    ch0_calculated_chA = [1,2,3]
    ch1_calculated_chA = [1,2,3]
    ch2_calculated_chA = [1,2,3]
    
    ch0_xyz_conversion_median = (np.median(ch0_alignment,axis=0).reshape(1,3))
    ch1_xyz_conversion_median = (np.median(ch1_alignment,axis=0).reshape(1,3))
    ch2_xyz_conversion_median = (np.median(ch2_alignment,axis=0).reshape(1,3))
    
    for conversion_i in range(0,3):
       
        ch0_calculated_chA[conversion_i] = ch0_xyz_conversion_median[0,conversion_i]  
        ch1_calculated_chA[conversion_i] = ch1_xyz_conversion_median[0,conversion_i]  
        ch2_calculated_chA[conversion_i] = ch2_xyz_conversion_median[0,conversion_i]
    
    
    shift_raw_ch0 = ch0_calculated_chA
    shift_raw_ch1 = ch1_calculated_chA
    shift_raw_ch2 = ch2_calculated_chA
    
    all_shift_raw = np.vstack((shift_raw_ch0,shift_raw_ch1,shift_raw_ch2))
    np.savetxt(path + "/"  + "all_shift_raw.csv",all_shift_raw,delimiter=",")
    
    all_cha_conversion = pd.DataFrame(all_cha_conversion )
    all_cha_conversion.to_csv(path + "/"  + "by_image_shift_raw.csv")     
        
          
            
            
            
            
            
            
            
#% 
    del psd_meta_data
    del syn_meta_data
    del all_data_variable
    del Normal_Freq
    del synaptic_loci
    del current_df

    

###################
## Default parameters
## Do not change
######################
running_channel_alignment = False
chA_image = False
percent = percent_main
chA_calculated = True
reference_channel_chA = []
current_channel_to_align = []
 #%
num_cores = mp.cpu_count()





text_to_save = [
'out_name =' + str(out_name) + '\n' +
'sequin_reference = ' + str(sequin_reference) + '\n' +
'sequin_partner = ' + str(sequin_partner) + '\n' +
'third_channel = ' + str(third_channel) + '\n' +
'cross_filter_spots =' + str(cross_filter_spots) + '\n' +
'three_channel_image =' + str(three_channel_image) + '\n' +
'percent_main =' + str(percent_main) + '\n' +
'percentile_intensity_cutoff =' + str(percentile_intensity_cutoff) + '\n' +
'gaussian_filter_image =' + str(gaussian_filter_image) + '\n' +
'gaussian =' + str(gaussian) + '\n' +
'z_intensity_correction_on =' + str(z_intensity_correction_on) + '\n' +
'pre_caluculated_intensity_cutoff =' + str(pre_caluculated_intensity_cutoff) + '\n' +
'image1_cutoff =' + str(image1_cutoff) + '\n' +
'image2_cutoff =' + str(image2_cutoff) + '\n' +
'synaptic_cutoff =' + str(synaptic_cutoff) + '\n' +
'voxelsize =' + str(voxelsize) + '\n' +
'manual_3rd_channel_bg_cutoff =' + str(manual_3rd_channel_bg_cutoff) + '\n' +
'third_channel_auto_BLS =' + str(third_channel_auto_BLS) + '\n' +
'third_channel_baseline_subtration =' + str(third_channel_baseline_subtration) + '\n' +
'quantile_filter_third_channel =' + str(quantile_filter_third_channel) + '\n' +
'quantile_cutoff_percent =' + str(quantile_cutoff_percent) + '\n' +
'synaptic_field_voxel_radius =' + str(synaptic_field_voxel_radius) + '\n' +
'neurite_field_voxel_radius =' + str(neurite_field_voxel_radius) + '\n' +
'rolling_ball_filtered =' + str(rolling_ball_filtered) + '\n' +
'nm_cutoff_all =' + str(nm_cutoff_all) + '\n' +
'auto_psd_syn_BLS =' + str(auto_psd_syn_BLS) + '\n' +
'channel_align_image =' + str(channel_align_image) + '\n' +
'percent_chA =' + str(percent_chA) + '\n' +
'reference_channel_chA =' + str(reference_channel_chA) + '\n' +
'channels_to_align =' + str(channels_to_align) + '\n' +
'shift_raw_ch0 =' + str(shift_raw_ch0) + '\n' +
'shift_raw_ch1 =' + str(shift_raw_ch1) + '\n' +
'shift_raw_ch2 =' + str(shift_raw_ch2) + '\n' +
'channel_alignment_override =' + str(channel_alignment_override) + '\n' +
'mannual_cha_ch0 =' + str(mannual_cha_ch0) + '\n' +
'mannual_cha_ch1 =' + str(mannual_cha_ch1) + '\n' +
'mannual_cha_ch2 ='+ str( mannual_cha_ch2) + '\n' +
'tile_scan_980 =' + str(tile_scan_980) + '\n' +
'pre_rolling_ball_intensity_filter =' + str(pre_rolling_ball_intensity_filter) + '\n' +
'pre_rb_wmbls1 =' + str(pre_rb_wmbls1) + '\n' +
'pre_rb_wmbls2 =' + str(pre_rb_wmbls2) + '\n' +
'wmbls1 =' + str(wmbls1) + '\n' +
'wmbls2 =' + str(wmbls2)+ '\n' +
'wmbls3 =' + str(wmbls3)]

save_df = pd.DataFrame(text_to_save)
save_df.to_csv(path + 'input_parameters.txt',header=False, index=False)

## Auto file key
image_position_key_auto = []
for image_key_i in range(0,len(files)):
    key_img = AICSImage(files[image_key_i],reconstruct_mosaic=False)
    current_img_tiles = np.shape(key_img)[0]
    for image_mosaic_tile_i in range(0,current_img_tiles):
        image_position_key_auto.append(np.hstack((image_key_i,image_mosaic_tile_i)))
image_position_key_auto=np.vstack(image_position_key_auto)


def perform_SEQUIN(row,current_channel_to_align,reference_channel_chA):
    
    ## Display must be confirmed to be turned off for code the process properly.
    display_detected_spots = False
    
    file_current_test = files[row[0]]
    mosaic_tile_i = row[1]
    
    
    img = AICSImage(file_current_test,reconstruct_mosaic=False) 
    lazy_t0 = img.get_image_dask_data("CZYX", M=mosaic_tile_i)  # returns out-of-memory 4D dask array
    image = lazy_t0.compute()

    
    if (chA_calculated == False) or (tile_scan_980 == False):
        tile_number = 0
        print('running this block')
        
        image1, image2, image3,third_channel_bg_cutoff = select_channels(image,shift_raw_ch0,shift_raw_ch1,shift_raw_ch2,spot_1_channel,spot_2_channel,spot_3_channel,current_channel_to_align,reference_channel_chA)
        
        image1 = np.where(image1>wmbls1,image1-wmbls1,0)
        image2 = np.where(image2>wmbls2,image2-wmbls2,0)
        image3 = np.where(image3>wmbls3,image3-wmbls3,0)
        
        
        for i in range (1,2):
            if i==1:
                qi = []
                
                image_name_raw = image1
                image_other_raw = image2
                file_name = file_current_test[0:-4] 
                name = file_name
                
                file_name = (name + 'channel_' + str(i))
                

                
                psd_meta_data, syn_meta_data,all_data_variable,Normal_Freq,synaptic_loci,current_df = spots_detection(image_name_raw,image_other_raw,file_name,out_name,path,i,qi,file_name,image3,tile_number,mosaic_tile_i,third_channel_bg_cutoff)
                
                
                
                del Normal_Freq
                del all_data_variable
                del current_channel_to_align
                del current_df
                del file_current_test
                del file_name
                del i
                del image
                del image1
                del image2
                del image3
                del image_name_raw
                del image_other_raw
                del img
                del lazy_t0
                del mosaic_tile_i
                del name
                del psd_meta_data
                del qi
                del reference_channel_chA
                del row
                del syn_meta_data
                del synaptic_loci
                del third_channel_bg_cutoff
                del tile_number
                gc.collect()
                save_df_variables = pd.DataFrame(dir())
                save_df_variables.to_csv(path + '_' + '_after_delete_remaining_variables.txt',header=False, index=False)
               
if display_detected_spots == True:
    image_to_display = image_position_key_auto[image_to_display,:]
    perform_SEQUIN(image_to_display, current_channel_to_align, reference_channel_chA)
#%%
####################################################################
## Temporary procesing pause.
## Confirm all variables and parameters have been defined properly.
## Run this block to perform SEQUIN
###################################################################

###############################################################
## Files to process can be truncated to user defined subet.
## Useful for testing large Tilescans
##image_position_key_auto = image_position_key_auto[258::]     
##############################################################

## Confirmed number of jobs to run matches computer resources and needs.

number_of_simultaneous_jobs = 5

## Run this block to start the processing.    
set_loky_pickler('cloudpickle')    
Parallel(n_jobs=number_of_simultaneous_jobs, verbose=50)(delayed(perform_SEQUIN)(row, current_channel_to_align, reference_channel_chA)for row in image_position_key_auto)  


