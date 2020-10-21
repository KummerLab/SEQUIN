# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:33:53 2020

@author: andrew.sauerbeck
"""

import pandas  as pd
import numpy as np
import skimage.feature as feature
import skimage.filters as filters
import time
import czifile as czi
import skimage.measure as measure
import skimage.segmentation as seg
import os
import re
import multiprocessing as mp
from joblib import Parallel, delayed
from sys import getsizeof
#%%


#Set input/output folders and naming conventions
#Input
folder = 'Y:/Active/2019 Airyscan/11-23-19old c57 section 5 right/AS6/AS6/'

files = np.sort(os.listdir(folder))




file = files[0]

if file == 'Thumbs.db':
        files = np.delete(files,0,0)
file = files[0]

name_start ='scan(.*)_Out'


#Set WM baseline subtraction value if using
wmbls1 = 588 #458
wmbls2 = 311 #280

#Define radius for weighted centroid calculation
radius =5
data_storage = []
gaussian = 1.76
cutoff = 0.7
variable = 10 # 10=peak13-mean,14-quantile. User sets which variable to use in finding connected voxels.
quantileP = .75

image = czi.imread(folder + '/' + file)


#Output
outputfolder = 'Y:/Active/2019 Airyscan/11-23-19old c57 section 5 right/AS6/Python_Out'
outfolder = "R5_C70_peakintensity_G1_76"
path = os.path.join(outputfolder, outfolder)



if os.path.exists(path) == False:
    os.mkdir(path)
    
out_name = (outfolder)

##User needs to check lines 366 and 367 to ensure channels are pulled correctly.

#%%
 

def spots_detection(image_name,image_name_raw,file_name,image_other_raw,out_name,path):
       
    tic = time.perf_counter()
    #Perform two levels of gaussian smoothing on image
    imageGauss1 = filters.gaussian(image_name, sigma=gaussian)
    #imageGauss2 = filters.gaussian(image_name, sigma=4)
    
    #Perform difference of gaussian subtraction and set negatives to zero
    #imageGauss = imageGauss1 - imageGauss2
    #imageGauss = np.where(imageGauss > 0,imageGauss,0)
    
    #Find local maxima
    peaklocal = feature.peak_local_max(imageGauss1,indices=True)
    #peaklocal = extrema.local_maxima(imageGauss1,indices=True)
    peaklocal = peaklocal.astype(np.int16)
    number_peaks = np.ma.size(peaklocal,axis=0)
    #del imageGauss
    
    #Skip images with none or too many spots
    if number_peaks > 0 and number_peaks < 220000:
               
        #create ID for each maxima to create labeled image
        indices = np.indices((number_peaks,2))
        indices = indices[0,:,:]
        indices = indices + 1
           
        
        # Create integers for coordinate adjustement to set maximum possible spot size
        coords = np.linspace(-radius,radius,num=((radius*2)+1),dtype=int)
        coords = coords.astype(np.int16)
        pairs = np.asarray(np.meshgrid(coords,coords,coords)).T.reshape(-1,3)
        X= np.sqrt((pairs[:,0]**2)+(pairs[:,1]**2)+(pairs[:,2]**2))
        #Select only spots within a max distance in pixels from centroid
        keep = pairs[X<=radius]
        number_keep = np.ma.size(keep,axis=0)
        
        del coords
        del pairs
        del X
                
        # Dupliacte peaks and adjustments and combine
    
        # Repeats each row X number of times sequentially
        total_points = np.repeat(peaklocal,number_keep,axis=0)
        total_indices = np.repeat(indices,number_keep,axis=0) 

    
        #Repeats data but keeps original sequence intact           
        total_keep = np.tile(keep,(number_peaks,1))
        
        #Combine arrays to get all indices within maximum radius from local maxima
        total_total = total_points + total_keep
        total_total = np.append(total_total,total_points,axis=1)
        total_total = np.append(total_total,total_indices,axis=1)
        
        
        
        df = pd.DataFrame(total_total)
        del total_points
        del total_indices           
        del total_keep
        del total_total 
        #del peaklocal
        del number_keep
        del keep

        # Remove adjusted coordinates outside of image border
    
        X = image_name.shape[2]
        Y = image_name.shape[1]
        Z = image_name.shape[0]
        
       
        
        df[6] = np.where(df[0] <0 , 0, 1)   #and df[:,0]< 1000
        df[6] = np.where(df[0] >(Z-1) , 0, 1) + df[6]  #and df[:,1]< 1000
        df[6] = np.where(df[1] <0 , 0, 1) + df[6]  #and df[:,0]< 1000
        df[6] = np.where(df[1] >(Y-1) , 0, 1) + df[6]  #and df[:,1]< 1000
        df[6] = np.where(df[2] <0 , 0, 1) + df[6]  #and df[:,0]< 1000
        df[6] = np.where(df[2] >(X-1) , 0, 1) + df[6]  #and df[:,1]< 1000
        df = df[df[6]==6]
            
        
        # Create column placeholders
        df.loc[:,8] = df[6]
        df.loc[:,9] = df[6]
        
        #Split center and radius positions
        dfarray = np.asarray(df)
        radiusZYXarray = dfarray[:,[0, 1, 2]]
        centerZYXarray = dfarray[:,[3, 4, 5]]
        del dfarray
        
        # Fill in row-wise center and radius intensity values
        df.loc[:,8] = image_name[(radiusZYXarray[:,0],radiusZYXarray[:,1],radiusZYXarray[:,2])]
        df.loc[:,9] = image_name[(centerZYXarray[:,0],centerZYXarray[:,1],centerZYXarray[:,2])]
        del radiusZYXarray
        del centerZYXarray
        
        # Perform connected threshold and keep voxels greater than 70% maxima intensity
        # Consider more sophisticated intensity selection alogrithm
        # Add column with distance from center location
        df.loc[:,10] = df.iloc[:,8] / df.iloc[:,9]
        
        voxels = df[[7,8]]
        means = voxels.groupby([7]).mean()
        quantile = voxels.groupby([7]).quantile(q=quantileP)
        df = df.merge(means,left_on=[7],right_index=True,how='inner')
        df = df.merge(quantile,left_on=[7],right_index=True,how='inner')
        
        df.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
        
        
        #df = np.asarray(df)
        #df = df.astype(np.int16)
        #df = pd.DataFrame(df)
        #df  = df.astype({'8':np.int16,'9':np.int16})
        df.loc[:,13] = df.iloc[:,8] / df.iloc[:,11]
        df.loc[:,14] = df.iloc[:,8] / df.iloc[:,12]
                
        df = df[df.iloc[:,variable]>cutoff] #Connected Threshold 10-fixed percent, 13-mean, 14-q uantile
        
        #Drop all rows with indices belonging to two peak centers
        #Makes assumption one can not tell which voxel should correspond to a given spot
        #Other options available        
        dropped=df.drop_duplicates(subset=['0','1','2'], inplace=False, keep=False)
        #number_df = np.ma.size(df,axis=1)
        #number_dropped = np.ma.size(dropped,axis=1)
        #number_dropped2 = number_df - number_dropped
        #number_dropped2 = number_dropped2 / number_df
        #data_out = np.hstack([number_df, number_dropped,number_dropped2]) 
        #data_storage.append(number_dropped2)
        
        # if i == 1:
        #     df_vs_dropped[nameint,0] = number_df
        #     df_vs_dropped[nameint,1] = number_dropped
        #     df_vs_dropped[nameint,2] = number_dropped2
        # if i == 2:
        #     df_vs_dropped[nameint,3] = number_df
        #     df_vs_dropped[nameint,4] = number_dropped
        #     df_vs_dropped[nameint,5] = number_dropped2
            
        
                
        del df
        
        #Create placeholder array of zeros
        fill = np.zeros((Z,Y,X))
        fill[(dropped.iloc[:,0],dropped.iloc[:,1],dropped.iloc[:,2])] = dropped.iloc[:,7]
        fill = fill.astype(int)
        
        #Remove objects touching border since maxima near border wouldn't have all voxels
        #Can leave pieces behind if object is not fully connected
        removed = seg.clear_border(fill)
        del fill
        
        #Check number of labels and skip image if no objects
        labels = np.unique(removed)
        labels = np.max(labels)
        if labels > 0:
            #Remove objects smaller than cutoff. If pure ellipse 30 corresponds to about 160nm by 788nm
            #props = ('label','area') 
            #table = measure.regionprops_table(label_image=removed, intensity_image=image_name_raw,properties=props) 
            #data = pd.DataFrame.from_dict(table, orient='columns')
            
            
            #small_objects = data[data['area']<0]
            #small_objects = np.asarray(small_objects)
            #small_objects = small_objects[:,0]
            
            #dropped = dropped[~dropped[7].isin(small_objects)]
            
            #unique_removed = np.unique(removed)
            #dropped = dropped[dropped[7].isin(unique_removed)]
            
            #fill = np.zeros((Z,Y,X))
            #fill[(dropped[0],dropped[1],dropped[2])] = dropped[7]
            #removed = fill.astype(int)
            del dropped
            #del fill
            #labels = np.unique(removed)
            #labels = np.max(labels)   
 
            if labels > 0:
                #Extract metadata based upon raw unprocessed image spots were created on
                props = ('label','area','mean_intensity','weighted_centroid','min_intensity',
                         'max_intensity','major_axis_length','minor_axis_length','centroid',
                         'equivalent_diameter','euler_number','extent','filled_area',
                         'inertia_tensor_eigvals') #'convex_area',,'solidity'
                #props = ('moments')#,'weighted_moments'   
                table1 = measure.regionprops_table(label_image=removed, intensity_image=image_name_raw,properties=props)
                table1 = np.asarray(pd.DataFrame.from_dict(table1, orient='columns'))
                
                
                table1df = pd.DataFrame(table1)

                table1df.loc[:,22] = table1df[0]
                table1df.loc[:,23] = table1df[0]
                table1df.loc[:,24] = table1df[0]
                
                indice = table1[:,0].astype(np.int16)
                indice = indice - 1
                table1df.loc[:,22:24] = peaklocal[indice[:]]
                
               
                #Extract metadata based upon raw unprocessed image of other channel
                props = ('mean_intensity','min_intensity','max_intensity')
                table2 = measure.regionprops_table(label_image=removed, intensity_image=image_other_raw,properties=props) 
                table2 = np.asarray(pd.DataFrame.from_dict(table2, orient='columns'))
                
                
                properties = measure.regionprops(label_image=removed, intensity_image=image_name_raw)# 
                del removed 
                
                           
                
                inertia_tensor_table = []
                moments_table = []
                moments_central_table = []
                moments_normalized_table = []
                weighted_moments_table = []
                weighted_moments_central_table = []
                weighted_moments_normalized_table = []
                    
                for row in properties:
                    inertia_tensor = np.ravel(row.inertia_tensor).reshape(9,1).T
                    inertia_tensor_table.append(inertia_tensor) 
                    
                    moments = np.ravel(row.moments).reshape(64,1).T
                    moments_table.append(moments)
                    
                    moments_central = np.ravel(row.moments_central).reshape(64,1).T
                    moments_central_table.append(moments_central)
                    
                    moments_normalized = np.ravel(row.moments_normalized).reshape(64,1).T
                    moments_normalized_table.append(moments_normalized)
                    
                    weighted_moments = np.ravel(row.weighted_moments).reshape(64,1).T
                    weighted_moments_table.append(weighted_moments)
                    
                    weighted_moments_central = np.ravel(row.weighted_moments_central).reshape(64,1).T
                    weighted_moments_central_table.append(weighted_moments_central)
                    
                    weighted_moments_normalized = np.ravel(row.weighted_moments_normalized).reshape(64,1).T
                    weighted_moments_normalized_table.append(weighted_moments_normalized)
                    
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
                
      
               
      
                                
                                   
                
                all_data = np.hstack([table1df, table2, inertia_tensor_table, moments_table, moments_central_table, moments_normalized_table, weighted_moments_table, weighted_moments_central_table, weighted_moments_normalized_table]) 
                np.savetxt(path + "/" + file_name + out_name  + ".csv",all_data,delimiter=",") #+ str(number_dropped)
                #np.savetxt(path + "/" + file_name + out_name  + "data_dropped.csv",data_out,delimiter=",") #+ str(number_dropped)
            else:
                X = np.zeros((1,10))
                np.savetxt(path + "/" + file_name + out_name +"_No_Spots.csv",X,delimiter=",")
        else:
            X = np.zeros((1,10))
            np.savetxt(path + "/" + file_name + out_name +"_No_Spots.csv",X,delimiter=",")
    else:
        X = np.zeros((1,10))
        np.savetxt(path + "/" + file_name + out_name +"_No_Spots.csv",X,delimiter=",")
    return #table1, table2, weighted_centroid

    toc = time.perf_counter()
    print("processtime: ")
    print(f"time {toc - tic:0.4f} seconds")
    

def test(row):
    #Select current czi
    file = row
    #Read in current czi and select channels
    image = czi.imread(folder + '/' + file)
    image1 = image[0,0,0,0,:,:,:,0]
    image2 = image[0,0,1,0,:,:,:,0] #changes to work with new images
    #Create new baseline subtratction images
    image1filtered = np.where(image1>wmbls1,image1-wmbls1,0)
    image2filtered = np.where(image2>wmbls2,image2-wmbls2,0)
    del image
    #Perform channel 1 and then channel 2 spots detection 
    for i in range (1,3):
        if i==1:
            image_name = image1filtered
            image_name_raw = image1
            image_other_raw = image2
        
        elif i==2:
            image_name = image2filtered
            image_name_raw = image2
            image_other_raw =image1
        file_name = file[0:-4] #re.search(name_start, file)
        name = file_name#.group(1)
        #nameint = int(name[1:4])
        file_name = (name + 'channel_' + str(i))#(file_name.group(1) + 'channel_' + str(i))
        spots_detection(image_name,image_name_raw,file_name,image_other_raw,out_name,path)


num_cores = mp.cpu_count()
Parallel(n_jobs=8, verbose=50)(delayed(test)(row)for row in files)    

