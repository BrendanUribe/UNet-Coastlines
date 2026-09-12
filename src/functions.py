import os
from sys import platform
import subprocess
from cmath import pi
from re import X
import shutil
from matplotlib import cm
import matplotlib.image as mpimg
import warnings
# # from sympy import Q
# # from vapory import *
from numpy.linalg import *
import matplotlib
import matplotlib.pyplot as plt
import cv2
import spiceypy as spice  # Spice library for computation of orbital elements
import math
import numpy as np
import timeit
import time
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Ellipse
from scipy import ndimage
from scipy import stats
from scipy.special import gamma
from PIL import Image
import pandas as pd
import csv
from skimage.util import random_noise
from skimage.restoration import estimate_sigma
# #HORIZONS APP: https://ssd.jpl.nasa.gov/horizons/app.html#/
kernels_path = os.path.join(os.getcwd(), "Kernels")
spice.furnsh(os.path.join(kernels_path, "naif0012.tls"))  # Time kernel
spice.furnsh(os.path.join(kernels_path, "de441.bsp"))  # Ephemeris kernel
spice.furnsh(os.path.join(kernels_path, "pck00010.tpc"))  # Frames kernel
# Frames kernel
spice.furnsh(os.path.join(kernels_path, "moon_pa_de403_1950-2198.bpc"))
spice.furnsh(os.path.join(kernels_path, "moon_060721.tf"))  # Frames kernel
spice.furnsh(os.path.join(kernels_path, "earth_latest_high_prec.bpc"))  # Frames kernel
import geopandas as gpd

def number2month(number):
    if number==1:
        month = 'JAN'
    elif number==2:
        month = 'FEB'
    elif number==3:
        month = 'MAR'
    elif number==4:
        month = 'APR'
    elif number==5:
        month = 'MAY'
    elif number==6:
        month = 'JUN'
    elif number==7:
        month = 'JUL'
    elif number==8:
        month = 'AUG'
    elif number==9:
        month = 'SEP'
    elif number==10:
        month = 'OCT'
    elif number==11:
        month = 'NOV'
    elif number==12:
        month = 'DEC'
    return month


def coastline_func_real(earth_map, cloud_map, coverage_threshold):
    
    # Load Earth map and cloud map
    map_earth = cv2.imread(earth_map)
    clouds = cv2.imread(cloud_map)
    
    # Resize the cloud map to match the Earth map's resolution
    clouds_resized = cv2.resize(clouds, (map_earth.shape[1], map_earth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Load coastline coordinates
    # coords = np.loadtxt('coordinates.csv', delimiter=',')
    coords = gpd.read_file("ne_10m_coastline.shp")  # 1:10m = high resolution
    x, y = coords[:, 0], coords[:, 1]
    
    # Convert from angles to pixel coordinates
    numy, numx, _ = map_earth.shape
    xpix = numx / 2 + 0.5 + (x / np.pi * numx / 2)
    ypix = numy / 2 + 0.5 - (y * 2 / np.pi * numy / 2)
    xypix = np.column_stack((xpix, ypix))

    # Load points of interest
    points = pd.read_csv('digitalized_points.csv').to_numpy()

    # Define thresholds
    # dist_threshold = 200  # in pixels
    dist_threshold = 300  # in pixels
    # coverage_threshold = 0.4  # visibility threshold for cloud cover

    # Initialize lists for bounding boxes and their lengths
    bounding_boxes_matrix = []
    filtered_bounding_boxes_matrix = []
    matrix_lengths = []
    filtered_boxes_info = np.empty((0, 8))
    counter_visible = 0
    coverage_list = []
    threshold_similar_rad = 10*np.pi/180

    # Iterate over each point of interest along the coastline
    for idx_point,point in enumerate(points):
        # Compute distances from each coastline point to the current digitalized point
        dist_point = np.sqrt(np.sum((xypix - point) ** 2, axis=1))
        idx_select = np.where(dist_point < dist_threshold)[0]

        if len(idx_select) == 0:
            continue

        # Select coastline points within the threshold
        xpix_select = xpix[idx_select]
        ypix_select = ypix[idx_select]

        x_min, x_max = np.min(xpix_select), np.max(xpix_select)
        y_min, y_max = np.min(ypix_select), np.max(ypix_select)
        x_half, y_half = (x_min + x_max) / 2, (y_min + y_max) / 2

        for case_id in range(9):
            if case_id == 0: #full box
                sub_idx = 0
                x_sub, y_sub = xpix_select, ypix_select
            elif case_id == 1: #top half           
                sub_idx = 1
                x_sub, y_sub = xpix_select[ypix_select <= y_half], ypix_select[ypix_select <= y_half]
            elif case_id == 2: #bottom half             
                sub_idx = 1
                x_sub, y_sub = xpix_select[ypix_select >= y_half], ypix_select[ypix_select >= y_half]
            elif case_id == 3: #right half              
                sub_idx = 1
                x_sub, y_sub = xpix_select[xpix_select >= x_half], ypix_select[xpix_select >= x_half]
            elif case_id == 4: #left half             
                sub_idx = 1
                x_sub, y_sub = xpix_select[xpix_select <= x_half], ypix_select[xpix_select <= x_half]
            elif case_id == 5: #top right
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select >= x_half) & (ypix_select <= y_half)], ypix_select [(xpix_select >= x_half) & (ypix_select <= y_half)]
            elif case_id == 6: #top left
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select <= x_half) & (ypix_select <= y_half)], ypix_select [(xpix_select <= x_half) & (ypix_select <= y_half)]
            elif case_id == 7: #bottom right
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select >= x_half) & (ypix_select >= y_half)], ypix_select [(xpix_select >= x_half) & (ypix_select >= y_half)]
            elif case_id == 8: #bottom left
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select <= x_half) & (ypix_select >= y_half)], ypix_select [(xpix_select <= x_half) & (ypix_select >= y_half)]

            if len(x_sub) == 0:
                continue
            
            # Cloud cover calculation for these selected points
            roundx = np.round(x_sub - 1).astype(int)
            roundy = np.round(y_sub - 1).astype(int)
            roundx = np.clip(roundx, 0, clouds_resized.shape[1] - 1)
            roundy = np.clip(roundy, 0, clouds_resized.shape[0] - 1)

            clouds_box = clouds_resized[roundy, roundx]
            average_coverage = np.mean(clouds_box) / 255
            visible = 1 if average_coverage < coverage_threshold else 0

            # Convert the selected pixel coordinates to radians.
            # For x: subtract half the image width and scale by 2*pi/width.
            # For y: subtract from half the image height and scale by (pi/2)/(half-height)
            x_rad_select = (x_sub - (numx / 2 + 0.5)) * (np.pi * 2 / numx)
            y_rad_select = ((numy / 2 + 0.5) - y_sub) * (np.pi / 2 / (numy / 2))

            len_select = len(x_rad_select)

            # Build the box matrix for these selected coastline points:
            # each row: [x_rad, y_rad, visible]
            Center_x, Center_y = [0.5 * (np.min(x_rad_select) + np.max(x_rad_select))], [
                0.5 * (np.min(y_rad_select) + np.max(y_rad_select))]
            box_matrix = np.column_stack((x_rad_select, y_rad_select, np.full(len_select, visible),
                                          np.full(len_select, Center_x), np.full(len_select, Center_y),
                                          np.full(len_select, case_id)))

            # Save the box and its length (number of points)
            bounding_boxes_matrix.append(box_matrix)
            matrix_lengths.append(len_select)

            # Only add visible bounding boxes to the filtered list
            if visible == 1:
                idx_allboxes = idx_point*9+case_id
                min_x,max_x,min_y,max_y = np.min(x_rad_select),np.max(x_rad_select),np.min(y_rad_select),np.max(y_rad_select)
                if counter_visible==0:                    
                    filtered_bounding_boxes_matrix.append(box_matrix)
                    filtered_boxes_info = np.hstack((len_select,sub_idx,idx_allboxes,idx_point,min_x,max_x,min_y,max_y,case_id))                    
                    coverage_list.append(average_coverage)
                    counter_visible+=1
                else:
                    if counter_visible==1:                  
                        differences_minmax = np.sum(np.abs(filtered_boxes_info[4:8]-np.array((min_x,max_x,min_y,max_y))))
                    else:
                        differences_minmax = np.sum(np.abs(filtered_boxes_info[:,4:8]-np.array((min_x,max_x,min_y,max_y))),axis=1)
                    if not np.any(differences_minmax<threshold_similar_rad):                        
                        filtered_bounding_boxes_matrix.append(box_matrix)
                        filtered_boxes_info = np.vstack((filtered_boxes_info,np.hstack((len_select,sub_idx,idx_allboxes,idx_point,min_x,max_x,min_y,max_y,case_id))))
                        counter_visible+=1
                        coverage_list.append(average_coverage)

    # For saving, we flatten the list of arrays into one array per file.
    if bounding_boxes_matrix:
        all_boxes = np.vstack(bounding_boxes_matrix)
    else:
        all_boxes = np.empty((0, 3))

    if filtered_bounding_boxes_matrix:
        all_filtered_boxes = np.vstack(filtered_bounding_boxes_matrix)
    else:
        all_filtered_boxes = np.empty((0, 3))
        
    # np.savetxt("bounding_boxes_matrix.txt", all_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility,Center_x,Center_y,case_id", comments="")
    # np.savetxt("filtered_bounding_boxes_matrix.txt", all_filtered_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility,Center_x,Center_y,case_id", comments="")
    # np.savetxt("matrix_lengths.txt", np.array(matrix_lengths).reshape(1, -1), delimiter=",",
    #            header="matrix_lengths", comments="")

    # Print messages confirming saves and show matrix lengths per bounding box
    # print("Bounding boxes matrix saved to bounding_boxes_matrix.txt")
    # print("Filtered bounding boxes matrix saved to filtered_bounding_boxes_matrix.txt")
    # print("Matrix lengths saved to matrix_lengths.txt")
    # print("Matrix Lengths (number of points per bounding box):")
    # print(matrix_lengths)
    # print(matrix_lengths_filt)

    filtered_boxes_array = np.vstack(filtered_bounding_boxes_matrix)
    geodlat_points = filtered_boxes_array[:, 1]
    lon_points = filtered_boxes_array[:, 0]
    cumulative_idx = np.hstack((0,np.cumsum(filtered_boxes_info[:-1,0]).flatten())).astype(int)
    x_centers = filtered_boxes_array[cumulative_idx, 3]
    y_centers = filtered_boxes_array[cumulative_idx, 4]

    a=6378.1
    b=6356.8

    e2 = 1-(b/a)**2 #Eccentricity
    N_points = a/np.sqrt(1-e2*np.sin(geodlat_points)**2)
    x_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.cos(lon_points)) #transpose, so each row represents one crater
    y_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.sin(lon_points))
    z_coord_rim = np.transpose((1-e2)*N_points*np.sin(geodlat_points))
    coordinates_rim = np.vstack((np.hstack((x_coord_rim.flatten())),
                                 np.hstack((y_coord_rim.flatten())), np.hstack((z_coord_rim.flatten())))) #[3x(num_points*num_craters)]

    N_points_centers = a / np.sqrt(1 - e2 * np.sin(y_centers) ** 2)
    x_center_boxes = np.transpose(
        N_points_centers * np.cos(y_centers) * np.cos(x_centers))  # transpose, so each row represents one crater
    y_center_boxes = np.transpose(N_points_centers * np.cos(y_centers) * np.sin(x_centers))
    z_center_boxes = np.transpose((1 - e2) * N_points_centers * np.sin(y_centers))
    center_boxes = np.vstack((np.hstack((x_center_boxes.flatten())),
                              np.hstack((y_center_boxes.flatten())),
                              np.hstack((z_center_boxes.flatten()))))
    
    coverage_list = np.array((coverage_list))
    total_points = len(points)

    return coordinates_rim, center_boxes, filtered_boxes_info, all_boxes, all_filtered_boxes, coverage_list, total_points


def coastline_func(earth_map, cloud_map, coverage_threshold):
    
    # Load Earth map and cloud map
    map_earth = cv2.imread(earth_map)
    clouds = cv2.imread(cloud_map)
    
    # Resize the cloud map to match the Earth map's resolution
    clouds_resized = cv2.resize(clouds, (map_earth.shape[1], map_earth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Load coastline coordinates
    # coords = np.loadtxt('coordinates.csv', delimiter=',')    
    coastline = gpd.read_file("ne_10m_coastline.shp")  # 1:10m = high resolution
    coords = []
    # Loop through all geometries
    for geom in coastline.geometry:
        if geom.geom_type == 'LineString':
            coords.extend(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords.extend(line.coords)

    # Convert to numpy array
    coords = np.array(coords)*np.pi/180  # shape (N, 2), with columns: [lon, lat]
    x, y = coords[:, 0], coords[:, 1]
    
    # Convert from angles to pixel coordinates
    numy, numx, _ = map_earth.shape
    xpix = numx / 2 + 0.5 + (x / np.pi * numx / 2)
    ypix = numy / 2 + 0.5 - (y * 2 / np.pi * numy / 2)
    xypix = np.column_stack((xpix, ypix))

    # Load points of interest
    points = pd.read_csv('digitalized_points.csv').to_numpy()

    # Define thresholds
    # dist_threshold = 200  # in pixels
    dist_threshold = 300  # in pixels
    # coverage_threshold = 0.4  # visibility threshold for cloud cover

    # Initialize lists for bounding boxes and their lengths
    bounding_boxes_matrix = []
    filtered_bounding_boxes_matrix = []
    matrix_lengths = []
    filtered_boxes_info = np.empty((0, 8))
    counter_visible = 0
    coverage_list = []
    threshold_similar_rad = 10*np.pi/180

    # Iterate over each point of interest along the coastline
    for idx_point,point in enumerate(points):
        # Compute distances from each coastline point to the current digitalized point
        dist_point = np.sqrt(np.sum((xypix - point) ** 2, axis=1))
        idx_select = np.where(dist_point < dist_threshold)[0]

        if len(idx_select) == 0:
            continue

        # Select coastline points within the threshold
        xpix_select = xpix[idx_select]
        ypix_select = ypix[idx_select]

        x_min, x_max = np.min(xpix_select), np.max(xpix_select)
        y_min, y_max = np.min(ypix_select), np.max(ypix_select)
        x_half, y_half = (x_min + x_max) / 2, (y_min + y_max) / 2

        # case_id = 0
        for case_id in range(9):
            if case_id == 0: #full box
                sub_idx = 0
                x_sub, y_sub = xpix_select, ypix_select
            elif case_id == 1: #top half           
                sub_idx = 1
                x_sub, y_sub = xpix_select[ypix_select <= y_half], ypix_select[ypix_select <= y_half]
            elif case_id == 2: #bottom half             
                sub_idx = 1
                x_sub, y_sub = xpix_select[ypix_select >= y_half], ypix_select[ypix_select >= y_half]
            elif case_id == 3: #right half              
                sub_idx = 1
                x_sub, y_sub = xpix_select[xpix_select >= x_half], ypix_select[xpix_select >= x_half]
            elif case_id == 4: #left half             
                sub_idx = 1
                x_sub, y_sub = xpix_select[xpix_select <= x_half], ypix_select[xpix_select <= x_half]
            elif case_id == 5: #top right
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select >= x_half) & (ypix_select <= y_half)], ypix_select [(xpix_select >= x_half) & (ypix_select <= y_half)]
            elif case_id == 6: #top left
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select <= x_half) & (ypix_select <= y_half)], ypix_select [(xpix_select <= x_half) & (ypix_select <= y_half)]
            elif case_id == 7: #bottom right
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select >= x_half) & (ypix_select >= y_half)], ypix_select [(xpix_select >= x_half) & (ypix_select >= y_half)]
            elif case_id == 8: #bottom left
                sub_idx = 2
                x_sub, y_sub = xpix_select[(xpix_select <= x_half) & (ypix_select >= y_half)], ypix_select [(xpix_select <= x_half) & (ypix_select >= y_half)]

            if len(x_sub) == 0:
                continue
            
            # Cloud cover calculation for these selected points
            roundx = np.round(x_sub - 1).astype(int)
            roundy = np.round(y_sub - 1).astype(int)
            roundx = np.clip(roundx, 0, clouds_resized.shape[1] - 1)
            roundy = np.clip(roundy, 0, clouds_resized.shape[0] - 1)

            clouds_box = clouds_resized[roundy, roundx]
            average_coverage = np.mean(clouds_box) / 255
            visible = 1 if average_coverage <= coverage_threshold else 0

            # Convert the selected pixel coordinates to radians.
            # For x: subtract half the image width and scale by 2*pi/width.
            # For y: subtract from half the image height and scale by (pi/2)/(half-height)
            x_rad_select = (x_sub - (numx / 2 + 0.5)) * (np.pi * 2 / numx)
            y_rad_select = ((numy / 2 + 0.5) - y_sub) * (np.pi / 2 / (numy / 2))

            len_select = len(x_rad_select)

            # Build the box matrix for these selected coastline points:
            # each row: [x_rad, y_rad, visible]
            Center_x, Center_y = [0.5 * (np.min(x_rad_select) + np.max(x_rad_select))], [
                0.5 * (np.min(y_rad_select) + np.max(y_rad_select))]
            box_matrix = np.column_stack((x_rad_select, y_rad_select, np.full(len_select, visible),
                                            np.full(len_select, Center_x), np.full(len_select, Center_y),
                                            np.full(len_select, case_id)))

            # Save the box and its length (number of points)
            bounding_boxes_matrix.append(box_matrix)
            matrix_lengths.append(len_select)

            # Only add visible bounding boxes to the filtered list
            if visible == 1:
                idx_allboxes = idx_point*9+case_id
                min_x,max_x,min_y,max_y = np.min(x_rad_select),np.max(x_rad_select),np.min(y_rad_select),np.max(y_rad_select)
                if counter_visible==0:                    
                    filtered_bounding_boxes_matrix.append(box_matrix)
                    filtered_boxes_info = np.hstack((len_select,sub_idx,idx_allboxes,idx_point,min_x,max_x,min_y,max_y,case_id))                    
                    coverage_list.append(average_coverage)
                    counter_visible+=1
                else:
                    if counter_visible==1:                  
                        differences_minmax = np.sum(np.abs(filtered_boxes_info[4:8]-np.array((min_x,max_x,min_y,max_y))))
                    else:
                        differences_minmax = np.sum(np.abs(filtered_boxes_info[:,4:8]-np.array((min_x,max_x,min_y,max_y))),axis=1)
                    if not np.any(differences_minmax<threshold_similar_rad):                        
                        filtered_bounding_boxes_matrix.append(box_matrix)
                        filtered_boxes_info = np.vstack((filtered_boxes_info,np.hstack((len_select,sub_idx,idx_allboxes,idx_point,min_x,max_x,min_y,max_y,case_id))))
                        counter_visible+=1
                        coverage_list.append(average_coverage)

    # For saving, we flatten the list of arrays into one array per file.
    if bounding_boxes_matrix:
        all_boxes = np.vstack(bounding_boxes_matrix)
    else:
        all_boxes = np.empty((0, 3))

    if filtered_bounding_boxes_matrix:
        all_filtered_boxes = np.vstack(filtered_bounding_boxes_matrix)
    else:
        all_filtered_boxes = np.empty((0, 3))
        
    # np.savetxt("bounding_boxes_matrix.txt", all_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility,Center_x,Center_y,case_id", comments="")
    # np.savetxt("filtered_bounding_boxes_matrix.txt", all_filtered_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility,Center_x,Center_y,case_id", comments="")
    # np.savetxt("matrix_lengths.txt", np.array(matrix_lengths).reshape(1, -1), delimiter=",",
    #            header="matrix_lengths", comments="")

    # Print messages confirming saves and show matrix lengths per bounding box
    # print("Bounding boxes matrix saved to bounding_boxes_matrix.txt")
    # print("Filtered bounding boxes matrix saved to filtered_bounding_boxes_matrix.txt")
    # print("Matrix lengths saved to matrix_lengths.txt")
    # print("Matrix Lengths (number of points per bounding box):")
    # print(matrix_lengths)
    # print(matrix_lengths_filt)

    filtered_boxes_array = np.vstack(filtered_bounding_boxes_matrix)
    geodlat_points = filtered_boxes_array[:, 1]
    lon_points = filtered_boxes_array[:, 0]
    cumulative_idx = np.hstack((0,np.cumsum(filtered_boxes_info[:-1,0]).flatten())).astype(int)
    x_centers = filtered_boxes_array[cumulative_idx, 3]
    y_centers = filtered_boxes_array[cumulative_idx, 4]

    # a=6378.137
    # b=6356.752314245
    a=6378.1
    b=6356.8

    e2 = 1-(b/a)**2 #Eccentricity
    N_points = a/np.sqrt(1-e2*np.sin(geodlat_points)**2)
    x_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.cos(lon_points)) #transpose, so each row represents one crater
    y_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.sin(lon_points))
    z_coord_rim = np.transpose((1-e2)*N_points*np.sin(geodlat_points))
    coordinates_rim = np.vstack((np.hstack((x_coord_rim.flatten())),
                                 np.hstack((y_coord_rim.flatten())), np.hstack((z_coord_rim.flatten())))) #[3x(num_points*num_craters)]

    N_points_centers = a / np.sqrt(1 - e2 * np.sin(y_centers) ** 2)
    x_center_boxes = np.transpose(
        N_points_centers * np.cos(y_centers) * np.cos(x_centers))  # transpose, so each row represents one crater
    y_center_boxes = np.transpose(N_points_centers * np.cos(y_centers) * np.sin(x_centers))
    z_center_boxes = np.transpose((1 - e2) * N_points_centers * np.sin(y_centers))
    center_boxes = np.vstack((np.hstack((x_center_boxes.flatten())),
                              np.hstack((y_center_boxes.flatten())),
                              np.hstack((z_center_boxes.flatten()))))
    
    coverage_list = np.array((coverage_list))
    total_points = len(points)

    return coordinates_rim, center_boxes, filtered_boxes_info, all_boxes, all_filtered_boxes, coverage_list, total_points

# ---------------------### Image Generation of Moon and Earth ###--------------------------
def coastline_func_old(earth_map,cloud_map,coverage_threshold):

    # Load Earth map and cloud map
    map_earth = cv2.imread(earth_map)
    clouds = cv2.imread(cloud_map)
    #
    # Resize the cloud map to match the Earth map's resolution
    clouds_resized = cv2.resize(clouds, (map_earth.shape[1], map_earth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Load coastline coordinates
    coords = np.loadtxt('coordinates.csv', delimiter=',')
    x, y = coords[:, 0], coords[:, 1]

    # Convert from angles to pixel coordinates
    numy, numx, _ = map_earth.shape
    xpix = numx / 2 + 0.5 + (x / np.pi * numx / 2)
    ypix = numy / 2 + 0.5 - (y * 2 / np.pi * numy / 2)
    xypix = np.column_stack((xpix, ypix))

    # Load points of interest
    points = pd.read_csv('digitalized_points.csv').to_numpy()

    # Define thresholds
    dist_threshold = 200  # in pixels
    # coverage_threshold = 0.4  # visibility threshold for cloud cover

    # Initialize lists for bounding boxes and their lengths
    bounding_boxes_matrix = []  # each element will be an array of shape (N, 3)
    filtered_bounding_boxes_matrix = []  # only for boxes with visible==1
    matrix_lengths = []  # store number of points per bounding box
    matrix_lengths_filt = []  # store number of points per filtered bounding box
    coverage_list = []

    # Iterate over each point of interest along the coastline
    for idx_point, point in enumerate(points):
        # Compute distances from each coastline point to the current digitalized point
        dist_point = np.sqrt(np.sum((xypix - point) ** 2, axis=1))
        idx_select = np.where(dist_point < dist_threshold)[0]

        if len(idx_select) == 0:
            continue

        # Select coastline points within the threshold
        xpix_select = xpix[idx_select]
        ypix_select = ypix[idx_select]

        # Cloud cover calculation for these selected points
        roundx = np.round(xpix_select - 1).astype(int)
        roundy = np.round(ypix_select - 1).astype(int)
        clouds_box = clouds_resized[roundy, roundx]
        average_coverage = np.mean(clouds_box) / 255
        visible = 1 if average_coverage <= coverage_threshold else 0

        # Convert the selected pixel coordinates to radians.
        # For x: subtract half the image width and scale by 2*pi/width.
        # For y: subtract from half the image height and scale by (pi/2)/(half-height).
        x_rad_select = (xpix_select - (numx / 2 + 0.5)) * (np.pi * 2 / numx)
        y_rad_select = ((numy / 2 + 0.5) - ypix_select) * (np.pi / 2 / (numy / 2))

        # Build the box matrix for these selected coastline points:
        # each row: [x_rad, y_rad, visible]
        Center_x, Center_y = [0.5*(np.min(x_rad_select)+np.max(x_rad_select))], [0.5*(np.min(y_rad_select)+np.max(y_rad_select))]
        box_matrix = np.column_stack((x_rad_select, y_rad_select, np.full(len(x_rad_select), visible), np.full(len(x_rad_select),Center_x), np.full(len(x_rad_select),Center_y)))#1/2*(np.min(y_rad_select)+np.max(y_rad_select))

        # Save the box and its length (number of points)
        bounding_boxes_matrix.append(box_matrix)
        matrix_lengths.append(len(box_matrix))

        # Only add visible bounding boxes to the filtered list
        if visible == 1:
            filtered_bounding_boxes_matrix.append(box_matrix)
            matrix_lengths_filt.append(len(box_matrix))
            coverage_list.append(average_coverage)

    # For saving, we flatten the list of arrays into one array per file.
    if bounding_boxes_matrix:
        all_boxes = np.vstack(bounding_boxes_matrix)
    else:
        all_boxes = np.empty((0, 3))

    if filtered_bounding_boxes_matrix:
        all_filtered_boxes = np.vstack(filtered_bounding_boxes_matrix)
    else:
        all_filtered_boxes = np.empty((0, 3))

    # np.savetxt("bounding_boxes_matrix.txt", all_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility", comments="")
    # np.savetxt("filtered_bounding_boxes_matrix.txt", all_filtered_boxes, delimiter=",",
    #            header="x_rad,y_rad,visibility", comments="")
    # np.savetxt("matrix_lengths.txt", np.array(matrix_lengths).reshape(1, -1), delimiter=",",
    #            header="matrix_lengths", comments="")

    # Print messages confirming saves and show matrix lengths per bounding box
    # print("Bounding boxes matrix saved to bounding_boxes_matrix.txt")
    # print("Filtered bounding boxes matrix saved to filtered_bounding_boxes_matrix.txt")
    # print("Matrix lengths saved to matrix_lengths.txt")
    # print("Matrix Lengths (number of points per bounding box):")
    # print(matrix_lengths)
    # print(matrix_lengths_filt)

    filtered_boxes_array = np.vstack(filtered_bounding_boxes_matrix)
    geodlat_points = filtered_boxes_array[:, 1]
    lon_points = filtered_boxes_array[:, 0]

    cumulative_idx = np.hstack((0,np.cumsum(matrix_lengths_filt[:-1]).flatten()))
    x_centers = filtered_boxes_array[cumulative_idx, 3]
    y_centers = filtered_boxes_array[cumulative_idx, 4]

    a=6378.1
    b=6356.8

    e2 = 1-(b/a)**2 #Eccentricity
    N_points = a/np.sqrt(1-e2*np.sin(geodlat_points)**2)
    x_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.cos(lon_points)) #transpose, so each row represents one crater
    y_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.sin(lon_points))
    z_coord_rim = np.transpose((1-e2)*N_points*np.sin(geodlat_points))
    coordinates_rim = np.vstack((np.hstack((x_coord_rim.flatten())),
                                 np.hstack((y_coord_rim.flatten())), np.hstack((z_coord_rim.flatten())))) #[3x(num_points*num_craters)]

    N_points_centers = a / np.sqrt(1 - e2 * np.sin(y_centers) ** 2)
    x_center_boxes = np.transpose(
        N_points_centers * np.cos(y_centers) * np.cos(x_centers))  # transpose, so each row represents one crater
    y_center_boxes = np.transpose(N_points_centers * np.cos(y_centers) * np.sin(x_centers))
    z_center_boxes = np.transpose((1 - e2) * N_points_centers * np.sin(y_centers))
    center_boxes = np.vstack((np.hstack((x_center_boxes.flatten())),
                              np.hstack((y_center_boxes.flatten())),
                              np.hstack((z_center_boxes.flatten()))))
    
    coverage_list = np.array((coverage_list))

    return coordinates_rim, center_boxes, matrix_lengths_filt, coverage_list

def coastline_mapping_vec_real(date, mission_specifics, camera_definition, xyz_moon, center_moon, fig3d_idx, plots_yesno,
                          test, num_points):  # points or ellipses
    # date, mission_specifics and camera_definition are from main.py and case_type function
    # mission_specifics = [theta,phi,dist,sc_xpos,sc_ypos,sc_zpos,moon_sp,moon_pos,sc_pos]
    # camera_definition = [look_at_object,focal_len,cam_width,cam_height,px]
    # type: 1 = ellipse, 2 = point
    sim_date = date
    # theta = mission_specifics[0]
    # phi = mission_specifics[1]
    # dist = mission_specifics[2]
    # sc_xpos = mission_specifics[3]
    # sc_ypos = mission_specifics[4]
    # sc_zpos = mission_specifics[5]
    # moon_sp = mission_specifics[6]
    # moon_pos = mission_specifics[7]
    earth_pos = np.array((0, 0, 0))
    sc_posx = mission_specifics[3]
    sc_posy = mission_specifics[4]
    sc_posz = mission_specifics[5]
    sc_pos = np.array((sc_posx,sc_posy,sc_posz))
    # print('moon_pos = ' + str(moon_pos))
    # print('sc_pos = ',str(sc_pos))
    # Position of the Sun compared to Earth in J2000 reference frame
    # sun = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [mission_specifics[13], mission_specifics[14], mission_specifics[15]]
    # ROTATIONS--------------------------------------------------------------------------
    quat = [mission_specifics[9],mission_specifics[10], mission_specifics[11], mission_specifics[12]]

    lat = mission_specifics[13]
    lon = mission_specifics[14]

    a=6378.1
    b=6356.8

    e2 = 1-(b/a)**2 #Eccentricity
    N_points = a/np.sqrt(1-e2*np.sin(lat)**2)
    x_centroid = np.transpose(N_points*np.cos(lat)*np.cos(lon)) #transpose, so each row represents one crater
    y_centroid = np.transpose(N_points*np.cos(lat)*np.sin(lon))
    z_centroid = np.transpose((1-e2)*N_points*np.sin(lat))

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "ITRF93", sim_date)

    r_centroid_J2000 = np.dot(np.transpose(
        rotm_moon), np.vstack((x_centroid,y_centroid,z_centroid))).flatten()

    # R_bi = rot_body2inertial(sc_pos, earth_pos)
    R_bi = quat2rot(quat[0],quat[1],quat[2],quat[3])
    
    roll = np.arctan2(R_bi[2,1],R_bi[2,2])*180/np.pi
    yaw = np.arctan2(R_bi[1,0],R_bi[0,0])*180/np.pi
    pitch = np.arctan2(-R_bi[2,0],R_bi[2,1]/np.sin(roll))*180/np.pi
    print("roll, pitch, yaw")
    print([roll, pitch, yaw])
    
    # R_bi = rpy_to_rot(-2.20671969, -0.50877685, 1.60669439)

    R_bi = rot_body2inertial([], [], rot_from_ME_to_J2000=np.transpose(rotm_moon), lineofsight=R_bi[:,0])
    print(R_bi)
    
    print(r_centroid_J2000)
    print(sc_pos)
    # R_bi = rot_body2inertial(sc_pos, earth_pos, rot_from_ME_to_J2000=np.transpose(rotm_moon))
    # print(R_bi)
  
    roll = np.arctan2(R_bi[2,1],R_bi[2,2])*180/np.pi
    yaw = np.arctan2(R_bi[1,0],R_bi[0,0])*180/np.pi
    pitch = np.arctan2(-R_bi[2,0],R_bi[2,1]/np.sin(roll))*180/np.pi
    print("roll, pitch, yaw")
    print([roll, pitch, yaw])

    craters_Moon_J200 = np.dot(np.transpose(
        rotm_moon), np.hstack((xyz_moon, center_moon)))  # concatenate center of the crater
    craters_earth = craters_Moon_J200 + np.vstack((earth_pos))

    craters_sc = craters_earth - np.vstack((sc_pos))
    craters_sun = craters_sc - np.vstack((sun_pos_e))
    craters_cam = np.dot(np.transpose(R_bi), craters_sc)
    # determining craters that face the camera
    craters_facing_cam = np.sum(craters_sc * craters_Moon_J200, axis=0)
    # determining which craters are light/dark
    craters_lit = np.sum(craters_sun * craters_Moon_J200, axis=0)
    num_pointscraters = int(np.sum(num_points))
    cumsum_points = np.cumsum(num_points).astype(int)
    num_boxes = len(num_points)
    craters_facing_cam_centers = craters_facing_cam[num_pointscraters:]
    craters_lit_centers = craters_lit[num_pointscraters:]
    craters_Moon_J200_centers = craters_Moon_J200[:, num_pointscraters:]
    craters_sc_centers = craters_sc[:, num_pointscraters:]
    craters_sun_centers = craters_sun[:, num_pointscraters:]
    # angle between crater center and sun vector
    angle_sun_deg = np.arccos(np.clip(-craters_lit_centers / np.sqrt(np.sum(
        craters_sun_centers ** 2, axis=0)) / np.sqrt(np.sum(craters_Moon_J200_centers ** 2, axis=0)), -1, 1)) * (
                                180 / np.pi)
    # spacecraft incidence angle
    #     sc_inc_angle_deg = np.arccos(np.min([1, np.max([-1, -craters_facing_cam[-1]/np.linalg.norm(craters_sc[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1])])]))*180/np.pi
    sc_inc_angle_deg = np.arccos(np.clip(-craters_facing_cam_centers / np.sqrt(np.sum(
        craters_sc_centers ** 2, axis=0)) / np.sqrt(np.sum(craters_Moon_J200_centers ** 2, axis=0)), -1, 1)) * (
                                   180 / np.pi)

    # Conversion to Pixels---------------------------------------
    # mm, #note, focal_len depends on case. Need to make this into function
    focal_len = int(camera_definition[1])
    px = float(camera_definition[4])
    cam_width = int(camera_definition[2])
    cam_height = int(camera_definition[3])

    X_c = craters_cam[0, :]
    Y_c = craters_cam[1, :]-focal_len/1000000
    Z_c = craters_cam[2, :]
    # another for loop
    pix_x = np.array(X_c) / np.array(Y_c) * focal_len / px
    pix_z = np.array(Z_c) / np.array(Y_c) * focal_len / px
    # plot
    plot_crater_x = pix_x + cam_width / 2
    plot_crater_z = cam_height / 2 - pix_z
    # Reshape: [num_craters x num_points]
    # plot_crater_x = np.hstack((np.reshape(plot_crater_x[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_x[num_pointscraters:]))))
    # plot_crater_z = np.hstack((np.reshape(plot_crater_z[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_z[num_pointscraters:]))))
    # craters_facing_cam = np.hstack((np.reshape(craters_facing_cam[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_facing_cam_centers))))
    # craters_lit = np.hstack((np.reshape(craters_lit[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_lit_centers))))
    # choose only values inside camera
    if test == 0:  # for training, points need to be within field of view, facing the camera, and illuminated
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
                plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0)
    elif test == 1:  # if testing, points do not need to be illuminated, only within the field of view and facing the camera
        # filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
        #         plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0)
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
                plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0)
    center_x = np.vstack((plot_crater_x[num_pointscraters:]))  # pixel of center
    center_z = np.vstack((plot_crater_z[num_pointscraters:]))
    center_visible = np.zeros((num_boxes, 1))
    center_visible[filter_visible[num_pointscraters:]] = 1
    visible = np.zeros((num_boxes, 1))
    for idx_box in range(num_boxes):
        if idx_box == 0:
            idx_first = 0
        else:
            idx_first = cumsum_points[idx_box - 1]
        sub_vec = filter_visible[idx_first:cumsum_points[idx_box]]
        if np.any(sub_vec):
            if np.all(sub_vec):
                visible[idx_box] = 2  # 2 if all points visible
            else:
                visible[idx_box] = 1  # 1 if partially visible

    # scale pixel values in case image is not the expected size
    plot_rim_x = plot_crater_x[:num_pointscraters]
    plot_rim_z = plot_crater_z[:num_pointscraters]
    isrim_visible = filter_visible[:num_pointscraters]
    # plotting
    # if plots_yesno == 1:
    #     for idx_box in range(num_boxes):
    #         visible_i = visible[idx_box]
    #         center_visible_i = center_visible[idx_box]
    #         if visible_i > 0:
    #             if idx_box == 0:
    #                 idx_first = 0
    #             else:
    #                 idx_first = cumsum_points[idx_box - 1]
    #             rim_select = isrim_visible[idx_first:cumsum_points[idx_box]]
    #             sub_vec_x = plot_rim_x[idx_first:cumsum_points[idx_box]]
    #             sub_vec_z = plot_rim_z[idx_first:cumsum_points[idx_box]]
    #             select_x = sub_vec_x[rim_select]
    #             select_z = sub_vec_z[rim_select]
    #             plt.figure(fig3d_idx, figsize=(15, 15))
    #             plt.plot(select_x, select_z, linestyle='None', marker='.', markersize=1.0)  # for 2D
    #             if center_visible_i == 1 and visible_i == 2:  # center is visible and whole rim as well
    #                 plt.plot(center_x[idx_box], center_z[idx_box], color='g', marker='x',
    #                          linestyle='None', markersize=1.0)
    #             elif center_visible_i == 1:  # center is visible but not the whole rim
    #                 plt.plot(center_x[idx_box], center_z[idx_box], color='r', marker='x',
    #                          linestyle='None', markersize=1.0)
    # # plt.savefig("moon_mapped.png",bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    return center_x, center_z, plot_rim_x, plot_rim_z, isrim_visible, center_visible, visible, angle_sun_deg, sc_inc_angle_deg


def coastline_mapping_vec(date, mission_specifics, camera_definition, xyz_moon, center_moon, fig3d_idx, plots_yesno,
                          test, num_points):  # points or ellipses
    # date, mission_specifics and camera_definition are from main.py and case_type function
    # mission_specifics = [theta,phi,dist,sc_xpos,sc_ypos,sc_zpos,moon_sp,moon_pos,sc_pos]
    # camera_definition = [look_at_object,focal_len,cam_width,cam_height,px]
    # type: 1 = ellipse, 2 = point
    sim_date = date
    # theta = mission_specifics[0]
    # phi = mission_specifics[1]
    # dist = mission_specifics[2]
    # sc_xpos = mission_specifics[3]
    # sc_ypos = mission_specifics[4]
    # sc_zpos = mission_specifics[5]
    # moon_sp = mission_specifics[6]
    # moon_pos = mission_specifics[7]
    earth_pos = np.array((0, 0, 0))
    sc_posx = mission_specifics[3]
    sc_posy = mission_specifics[4]
    sc_posz = mission_specifics[5]
    sc_pos = np.array((sc_posx,sc_posy,sc_posz))
    # print('moon_pos = ' + str(moon_pos))
    # print('sc_pos = ',str(sc_pos))
    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]
    # ROTATIONS--------------------------------------------------------------------------
    R_bi = rot_body2inertial(sc_pos, earth_pos)
    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "IAU_EARTH", sim_date)

    craters_Moon_J200 = np.dot(np.transpose(
        rotm_moon), np.hstack((xyz_moon, center_moon)))  # concatenate center of the crater
    craters_earth = craters_Moon_J200 + np.vstack((earth_pos))

    craters_sc = craters_earth - np.vstack((sc_pos))
    craters_sun = craters_sc - np.vstack((sun_pos_e))
    craters_cam = np.dot(np.transpose(R_bi), craters_sc)
    X_c = craters_cam[0, :]
    Y_c = craters_cam[1, :]
    Z_c = craters_cam[2, :]
    # determining craters that face the camera
    craters_facing_cam = np.sum(craters_sc * craters_Moon_J200, axis=0)
    # determining which craters are light/dark
    craters_lit = np.sum(craters_sun * craters_Moon_J200, axis=0)
    num_pointscraters = int(np.sum(num_points))
    cumsum_points = np.cumsum(num_points).astype(int)
    num_boxes = len(num_points)
    craters_facing_cam_centers = craters_facing_cam[num_pointscraters:]
    craters_lit_centers = craters_lit[num_pointscraters:]
    craters_Moon_J200_centers = craters_Moon_J200[:, num_pointscraters:]
    craters_sc_centers = craters_sc[:, num_pointscraters:]
    craters_sun_centers = craters_sun[:, num_pointscraters:]
    # angle between crater center and sun vector
    angle_sun_deg = np.arccos(np.clip(-craters_lit_centers / np.sqrt(np.sum(
        craters_sun_centers ** 2, axis=0)) / np.sqrt(np.sum(craters_Moon_J200_centers ** 2, axis=0)), -1, 1)) * (
                                180 / np.pi)
    # spacecraft incidence angle
    #     sc_inc_angle_deg = np.arccos(np.min([1, np.max([-1, -craters_facing_cam[-1]/np.linalg.norm(craters_sc[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1])])]))*180/np.pi
    sc_inc_angle_deg = np.arccos(np.clip(-craters_facing_cam_centers / np.sqrt(np.sum(
        craters_sc_centers ** 2, axis=0)) / np.sqrt(np.sum(craters_Moon_J200_centers ** 2, axis=0)), -1, 1)) * (
                                   180 / np.pi)

    # Conversion to Pixels---------------------------------------
    # mm, #note, focal_len depends on case. Need to make this into function
    focal_len = int(camera_definition[1])
    px = float(camera_definition[4])
    cam_width = int(camera_definition[2])
    cam_height = int(camera_definition[3])
    # another for loop
    pix_x = np.array(X_c) / np.array(Y_c) * focal_len / px
    pix_z = np.array(Z_c) / np.array(Y_c) * focal_len / px
    # plot
    plot_crater_x = pix_x + cam_width / 2
    plot_crater_z = cam_height / 2 - pix_z
    # Reshape: [num_craters x num_points]
    # plot_crater_x = np.hstack((np.reshape(plot_crater_x[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_x[num_pointscraters:]))))
    # plot_crater_z = np.hstack((np.reshape(plot_crater_z[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_z[num_pointscraters:]))))
    # craters_facing_cam = np.hstack((np.reshape(craters_facing_cam[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_facing_cam_centers))))
    # craters_lit = np.hstack((np.reshape(craters_lit[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_lit_centers))))
    # choose only values inside camera
    if test == 0:  # for training, points need to be within field of view, facing the camera, and illuminated
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
                plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0)
    elif test == 1:  # if testing, points do not need to be illuminated, only within the field of view and facing the camera
        # filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
        #         plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0)
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
                plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0)
    center_x = np.vstack((plot_crater_x[num_pointscraters:]))  # pixel of center
    center_z = np.vstack((plot_crater_z[num_pointscraters:]))
    center_visible = np.zeros((num_boxes, 1))
    center_visible[filter_visible[num_pointscraters:]] = 1
    visible = np.zeros((num_boxes, 1))
    for idx_box in range(num_boxes):
        if idx_box == 0:
            idx_first = 0
        else:
            idx_first = cumsum_points[idx_box - 1]
        sub_vec = filter_visible[idx_first:cumsum_points[idx_box]]
        if np.any(sub_vec):
            if np.all(sub_vec):
                visible[idx_box] = 2  # 2 if all points visible
            else:
                visible[idx_box] = 1  # 1 if partially visible

    # scale pixel values in case image is not the expected size
    plot_rim_x = plot_crater_x[:num_pointscraters]
    plot_rim_z = plot_crater_z[:num_pointscraters]
    isrim_visible = filter_visible[:num_pointscraters]
    # plotting
    # if plots_yesno == 1:
    #     for idx_box in range(num_boxes):
    #         visible_i = visible[idx_box]
    #         center_visible_i = center_visible[idx_box]
    #         if visible_i > 0:
    #             if idx_box == 0:
    #                 idx_first = 0
    #             else:
    #                 idx_first = cumsum_points[idx_box - 1]
    #             rim_select = isrim_visible[idx_first:cumsum_points[idx_box]]
    #             sub_vec_x = plot_rim_x[idx_first:cumsum_points[idx_box]]
    #             sub_vec_z = plot_rim_z[idx_first:cumsum_points[idx_box]]
    #             select_x = sub_vec_x[rim_select]
    #             select_z = sub_vec_z[rim_select]
    #             plt.figure(fig3d_idx, figsize=(15, 15))
    #             plt.plot(select_x, select_z, linestyle='None', marker='.', markersize=1.0)  # for 2D
    #             if center_visible_i == 1 and visible_i == 2:  # center is visible and whole rim as well
    #                 plt.plot(center_x[idx_box], center_z[idx_box], color='g', marker='x',
    #                          linestyle='None', markersize=1.0)
    #             elif center_visible_i == 1:  # center is visible but not the whole rim
    #                 plt.plot(center_x[idx_box], center_z[idx_box], color='r', marker='x',
    #                          linestyle='None', markersize=1.0)
    # # plt.savefig("moon_mapped.png",bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    return center_x, center_z, plot_rim_x, plot_rim_z, isrim_visible, center_visible, visible, angle_sun_deg, sc_inc_angle_deg


def find_x_limits(pos_cam, lookat_y, map_x, map_y, focal_len_mm, cam_width, cam_height, pixel_size_mm):
    cx = pos_cam[0]  # camera x coordinate, km
    cy = pos_cam[1]  # camera y coordinate, km
    cz = pos_cam[2]  # camera z coordinate (height), km

    dy = lookat_y-cy  # delta y from camera to look_at location, km
    dz = 0-cz  # delta z  from camera to look_at location, km
    # dx is unknown and we want to know its upper and lower limits

    # Equations for upper limit of dx
    cw2 = cam_width*pixel_size_mm*1e-6/2  # camera width over 2, km
    ch2 = cam_height*pixel_size_mm*1e-6/2  # camera height over 2, km
    f = focal_len_mm*1e-6  # camera focal length, km

    # Initialize Newton's method
    error_value = np.inf  # we want this value to converge to zero
    dx = 0  # initial guess for dx upper limit, km
    while error_value > 1e-9:  # iterate until a solution is found
        # norm of planar vector from camera to look_at location, function of dx
        norm_2D = np.sqrt(dx**2+dy**2)
        # norm of 3D vector from camera to look_at location, function of dx
        norm_3D = np.sqrt(dx**2+dy**2+dz**2)
        # multiplication factor to intersect edge of map, solved from: camera_location[2] + v_corner[2]*L = 0
        L = - cz/(f*dz + ch2*norm_2D)
        # this is the x-coordinate of the corner vector in inertial frame: v_corner[0]
        B = cw2*dy*norm_3D/norm_2D + f*dx - ch2*dx*dz/norm_2D
        # equation to solve: camera_location[0] + v_corner[0]*L = map_x/2 (intersection with "right" edge of map)
        value = cx - map_x/2 + B*L
        dvalue = (cw2*dy*dx*(1/(norm_2D*norm_3D) - norm_3D/norm_2D**3) + f - ch2 *
                  dz*(1/norm_2D - dx**2/norm_2D**3))*L + B*(cz/(f*dz + ch2*norm_2D)**2*ch2*dx/norm_2D)  # derivative of value wrt dx: dB*L + B*dL
        dx = dx - value/dvalue  # update dx based on current value and current derivative wrt dx
        error_value = np.abs(value)  # absolute value of error

    # once converged, compute upper limit of look_at location as camera_location+dx
    x_upper_limit = cx+dx

    # Equations for lower limit of dx (same as before but changing the sign of the corner vector, and of the edge that is intersected)
    # camera width over 2, km (now we want the intersection of extended top-left corner, hence minus sign)
    cw2 = -cam_width*pixel_size_mm*1e-6/2
    ch2 = cam_height*pixel_size_mm*1e-6/2  # camera height over 2, km
    f = focal_len_mm*1e-6  # camera focal length, km

    # Initialize Newton's method
    error_value = np.inf  # we want this value to converge to zero
    dx = 0  # initial guess for dx lower limit, km
    while error_value > 1e-9:  # iterate until a solution is found
        # norm of planar vector from camera to look_at location, function of dx
        norm_2D = np.sqrt(dx**2+dy**2)
        # norm of 3D vector from camera to look_at location, function of dx
        norm_3D = np.sqrt(dx**2+dy**2+dz**2)
        # multiplication factor to intersect edge of map, solved from: camera_location[2] + v_corner[2]*L = 0
        L = - cz/(f*dz + ch2*norm_2D)
        # this is the x-coordinate of the corner vector in inertial frame: v_corner[0]
        B = cw2*dy*norm_3D/norm_2D + f*dx - ch2*dx*dz/norm_2D
        # equation to solve: camera_location[0] + v_corner[0]*L = -map_x/2 (notice minus sign, since we want the intersection with the "left" edge of map)
        value = cx + map_x/2 + B*L
        dvalue = (cw2*dy*dx*(1/(norm_2D*norm_3D) - norm_3D/norm_2D**3) + f - ch2 *
                  dz*(1/norm_2D - dx**2/norm_2D**3))*L + B*(cz/(f*dz + ch2*norm_2D)**2*ch2*dx/norm_2D)  # derivative of value wrt dx: dB*L + B*dL
        dx = dx - value/dvalue  # update dx based on current value and current derivative wrt dx
        error_value = np.abs(value)  # absolute value of error

    # once converged, compute lower limit of look_at location as camera_location+dx
    x_lower_limit = cx+dx

    return x_upper_limit, x_lower_limit


def gen_plane(pos_cam, pos_lookat, camera_definition, iter, map_x, map_y):

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------
    pov_file = open('tmp_' + str(int(iter)) + '.pov',
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # ------------------------------------------------------------------------------------------------------
    pov_file.write(
        '#declare mapbox = box { <' + str(-map_x/2+1) + ',-10,' + str(-map_y/2+1) +
        '>, <' + str(map_x/2-1) + ',0,' + str(map_y/2-1) + '>\n'
        '                       texture { pigment { color rgb <1,1,1> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare maplinesx = box { <' + str(-map_x/2) + ',-0.1,' + str(-0.5) +
        '>, <' + str(map_x/2) + ',0,' + str(0.5) + '>\n'
        '                       texture { pigment { color rgb <0,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare maplinesz = box { <' + str(-0.5) + ',-0.1,' + str(-map_y/2) +
        '>, <' + str(0.5) + ',0,' + str(map_y/2) + '>\n'
        '                       texture { pigment { color rgb <0,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare mapedgez1 = box { <' + str(map_x/2-1) + ',-5,' + str(-map_y/2) +
        '>, <' + str(map_x/2) + ',0,' + str(map_y/2) + '>\n'
        '                       texture { pigment { color rgb <1,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare mapedgez2 = box { <' + str(-map_x/2) + ',-5,' + str(-map_y/2) +
        '>, <' + str(-map_x/2+1) + ',0,' + str(map_y/2) + '>\n'
        '                       texture { pigment { color rgb <1,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare mapedgex1 = box { <' + str(-map_x/2) + ',-5,' + str(-map_y/2) +
        '>, <' + str(map_x/2) + ',0,' + str(-map_y/2+1) + '>\n'
        '                       texture { pigment { color rgb <1,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')
    pov_file.write(
        '#declare mapedgex2 = box { <' + str(-map_x/2) + ',-5,' + str(map_y/2-1) +
        '>, <' + str(map_x/2) + ',0,' + str(map_y/2) + '>\n'
        '                       texture { pigment { color rgb <1,0,0> }}\n'
        '                       rotate -90*x scale <1,1,-1>}\n')

    pov_file.write('object {mapbox}\n')
    pov_file.write('object {maplinesx}\n')
    pov_file.write('object {maplinesz}\n')
    pov_file.write('object {mapedgez1}\n')
    pov_file.write('object {mapedgez2}\n')
    pov_file.write('object {mapedgex1}\n')
    pov_file.write('object {mapedgex2}\n')

    focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    image_name = "plane_img_" + str(int(iter)) + ".png"

    scene_file = os.path.join(os.getcwd(), image_name)
    scene_fileq = '"' + scene_file + '"'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    sky z\n'
                   '    direction z\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' +
                   str(pos_cam[0]) + ',' + str(pos_cam[1]) +
                   ',' + str(pos_cam[2]) + '>\n'
                   '    up z\n'
                   '    look_at <' +
                   str(pos_lookat[0]) + ',' + str(pos_lookat[1]) +
                   ',' + str(pos_lookat[2]) + '>\n'
                   '}\n')

    pov_file.write('light_source { \n'
                   '    <' + str(0) + ',' +
                   str(0) + ',' +
                   str(10000) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    ini_file = open('tmp_' + str(int(iter)) + '.ini',
                    'w')  # .ini temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        'Input_File_Name=' + '"' +
        os.path.join(os.getcwd(), "tmp_" +
                     str(int(iter)) + ".pov") + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(os.getcwd(), "tmp_" + str(int(iter)) + ".ini") + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        os.system(povray_location + ' "' +
                  os.path.join(os.getcwd(), "tmp_" + str(int(iter)) + ".ini") + '"')

    return scene_file


def gen_moon_earth(sc_eph, time_eph, camera_definition, iter, ocean_reflection, lookat_input=[], sky_vec_input=[], scenario=[], month_number=[], day=[], hour=[]):

    print('Start of image generation')
    start = timeit.default_timer()

    # determines month of the year -> changes earth map
    year = spice.timout(time_eph, 'YYYY')
    month = spice.timout(time_eph, 'MON')

    ### -------- INPUTS -------- ###
    # sc_eph:       spacecraft coordinates in J2000
    # time_eph:     ephemeris time
    # case:         Scenarios: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    # iter:         Iteration number for image generation purposes
    ### ------------------------ ###

    # Start of time count for the efficiency
    # print('Start of image generation')
    # start = timeit.default_timer()

    # Constants ------------------------------------------------------------------------------------------------------------------------------------------
    # km #scaling factor is: (variable in kilometers)/R_Earth*3.5
    R_Earth = 6378.1
    R_Earth_pole = 6356.8  # km
    highest_pt = 8.850  # km

    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    highest_pt = 11.000         # km

    # Calculations  ------------------------------------------------------------------------------------------------------------------------------------------
    # Position of the Moon compared to Earth in J2000 (ECI) reference frame
    moon = spice.spkpos("301", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    moon_pos_e = [moon[0][0], moon[0][1], moon[0][2]]

    # Position of the Spacecraft in the J2000 frame
    sc_pos = sc_eph

    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]

    # Position of Star Map: centered in Solar System Barycenter (spice ID: 0)
    star = spice.spkpos("0", time_eph, "J2000", "NONE", "399")
    star_pos_e = [star[0][0], star[0][1], star[0][2]]

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", time_eph)

    # Rotation of the Earth on its axis
    rotm_earth = spice.pxform("J2000", "IAU_EARTH", time_eph)

    # Visual Magnitude Parameters
    # Compute vector Earth to Sun
    earth2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    earth2sun = np.array([earth2sun[0][0], earth2sun[0][1], earth2sun[0][2]])
    # Compute the vector Moon to Sun
    moon2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "301")
    moon2sun = np.array([moon2sun[0][0], moon2sun[0][1], moon2sun[0][2]])
    # Compute the vector Earth to Spacecraft
    earth2sc = np.array(sc_pos)
    # Compute the vector Moon to Spacecraft
    moon2sc = np.array(sc_pos) - np.array(moon_pos_e)
    # Call function to calculate the Apparent Magnitude
    V_e, V_m, alpha_m = magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc)
    # print('The apparent magnitude of the Earth is: ', V_e)
    # print('The apparent magnitude of the Moon is: ', V_m)
    # print('The phase angle of the Moon is: '+str("{:.2f}".format(alpha_m))+' deg')

    #Cloud maps: https://www.shadedrelief.com/natural3/pages/clouds.html
    # Texture Maps Earth------------------------------------------------------------------------------------------------------------------------------------------
    if day.size == 0:
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/earth/clouds.jpg") + '"'
        cloud_map = []
    else:       
        if month_number<10:
            str_month = "0" + str(int(month_number))
        else:
            str_month = str(int(month_number))
        if day<10:
            str_day = "0" + str(int(day))
        else:
            str_day = str(int(day))
        if hour<10:
            str_hour = "0" + str(int(hour))
        else:
            str_hour = str(int(hour))
        cloud_map = str(year) + "-" + str_month + "-" + str_day + "T" + str_hour + "-00-00.000Z.jpg"
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/clouds/allyear/" + cloud_map) + '"'
    # color_star_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/TychoSkymapII_v2.tiff") + '"'
    # new_clouds = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/pngkit-cloud-texture-png-961670.tif") + '"'
    # new_land = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_land.tiff") + '"'
    # color_oceanblack = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_oceanblackcontinentwhite.tiff") + '"'

    # color_Earth_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/world_shaded_scaled.jpg") + '"'
    # cloud_fraction = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/cloud_fraction.jpg") + '"'

    # Texture Maps Moon
    # color_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_color_scaled.tiff") + '"'
    # # elevation_fileq = '"' + \
    # #     os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'
    # new_moon = '"' + os.path.join(os.getcwd(), "Maps/moon/new_moon.tiff") + '"'
    # moon_craters2 = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_craters2.tiff") + '"'

    # maps for each month (WITHOUT ocean)
    earth_jan_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jan_adj.jpg") + '"'
    earth_feb_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_feb_adj.jpg") + '"'
    earth_mar_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_mar_adj.jpg") + '"'
    earth_apr_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_apr_adj.jpg") + '"'
    earth_may_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_may_adj.jpg") + '"'
    earth_jun_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jun_adj.jpg") + '"'
    earth_jul_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jul_adj.jpg") + '"'
    earth_aug_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_aug_adj.jpg") + '"'
    earth_sept_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept_adj.jpg") + '"'
    earth_oct_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_oct_adj.jpg") + '"'
    earth_nov_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_nov_adj.jpg") + '"'
    earth_dec_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_dec_adj.jpg") + '"'

    # maps for each month (WITH ocean)
    earth_jan = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jan.jpg") + '"'
    earth_feb = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_feb.jpg") + '"'
    earth_mar = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_mar.jpg") + '"'
    earth_apr = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_apr.jpg") + '"'
    earth_may = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_may.jpg") + '"'
    earth_jun = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jun.jpg") + '"'
    earth_jul = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jul.jpg") + '"'
    earth_aug = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_aug.jpg") + '"'
    earth_sept = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept.jpg") + '"'
    earth_oct = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_oct.jpg") + '"'
    earth_nov = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_nov.jpg") + '"'
    earth_dec = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_dec.jpg") + '"'

    # color_ocean_black_continent_white
    color_oceanblack_jan = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jan.jpg") + '"'
    color_oceanblack_feb = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_feb.jpg") + '"'
    color_oceanblack_mar = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_mar.jpg") + '"'
    color_oceanblack_apr = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_apr.jpg") + '"'
    color_oceanblack_may = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_may.jpg") + '"'
    color_oceanblack_jun = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jun.jpg") + '"'
    color_oceanblack_jul = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jul.jpg") + '"'
    color_oceanblack_aug = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_aug.jpg") + '"'
    color_oceanblack_sept = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_sept.jpg") + '"'
    color_oceanblack_oct = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_oct.jpg") + '"'
    color_oceanblack_nov = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_nov.jpg") + '"'
    color_oceanblack_dec = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_dec.jpg") + '"'

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------

    # determine the month
    if month == 'JAN':
        earth_month = earth_jan
        earth_month_adj = earth_jan_adj
        color_oceanblack_month = color_oceanblack_jan
        earth_map = "earth_jan.jpg"
        earth_adj = "earth_jan_adj.jpg"
        print('JAN')
    elif month == 'FEB':
        earth_month = earth_feb
        earth_month_adj = earth_feb_adj
        color_oceanblack_month = color_oceanblack_feb
        earth_map = "earth_feb.jpg"
        earth_adj = "earth_feb_adj.jpg"
        print('FEB')
    elif month == 'MAR':
        earth_month = earth_mar
        earth_month_adj = earth_mar_adj
        color_oceanblack_month = color_oceanblack_mar
        earth_map = "earth_mar.jpg"
        earth_adj = "earth_mar_adj.jpg"
        print('MAR')
    elif month == 'APR':
        earth_month = earth_apr
        earth_month_adj = earth_apr_adj
        color_oceanblack_month = color_oceanblack_apr
        earth_map = "earth_apr.jpg"
        earth_adj = "earth_apr_adj.jpg"
        print('APR')
    elif month == 'MAY':
        earth_month = earth_may
        earth_month_adj = earth_may_adj
        color_oceanblack_month = color_oceanblack_may
        earth_map = "earth_may.jpg"
        earth_adj = "earth_may_adj.jpg"
        print('MAY')
    elif month == 'JUN':
        earth_month = earth_jun
        earth_month_adj = earth_jun_adj
        color_oceanblack_month = color_oceanblack_jun
        earth_map = "earth_jun.jpg"
        earth_adj = "earth_jun_adj.jpg"
        print('JUN')
    elif month == 'JUL':
        earth_month = earth_jul
        earth_month_adj = earth_jul_adj
        color_oceanblack_month = color_oceanblack_jul
        earth_map = "earth_jul.jpg"
        earth_adj = "earth_jul_adj.jpg"
        print('JUL')
    elif month == 'AUG':
        earth_month = earth_aug
        earth_month_adj = earth_aug_adj
        color_oceanblack_month = color_oceanblack_aug
        earth_map = "earth_aug.jpg"
        earth_adj = "earth_aug_adj.jpg"
        print('AUG')
    elif month == 'SEP':
        earth_month = earth_sept
        earth_month_adj = earth_sept_adj
        color_oceanblack_month = color_oceanblack_sept
        earth_map = "earth_sept.jpg"
        earth_adj = "earth_sept_adj.jpg"
        print('SEP')
    elif month == 'OCT':
        earth_month = earth_oct
        earth_month_adj = earth_oct_adj
        color_oceanblack_month = color_oceanblack_oct
        earth_map = "earth_oct.jpg"
        earth_adj = "earth_oct_adj.jpg"
        print('OCT')
    elif month == 'NOV':
        earth_month = earth_nov
        earth_month_adj = earth_nov_adj
        color_oceanblack_month = color_oceanblack_nov
        earth_map = "earth_nov.jpg"
        earth_adj = "earth_nov_adj.jpg"
        print('NOV')
    elif month == 'DEC':
        earth_month = earth_dec
        earth_month_adj = earth_dec_adj
        color_oceanblack_month = color_oceanblack_dec
        earth_map = "earth_dec.jpg"
        earth_adj = "earth_dec_adj.jpg"
        print('DEC')

# Other Objects of Scene()
    if scenario==[]:
        pov_name = "tmp_" + str(int(iter)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '.ini'
    else:
        pov_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.ini'
    pov_file = open(pov_namesingle,
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # pov_file.write('// GLOBAL SETTINGS\n'
    #         'global_settings { \n'
    #         '    assumed_gamma 1.0\n'
    #         '    radiosity { \n'
    #         '        count 1000\n'
    #         '        nearest_count 20\n'
    #         '        recursion_limit 1\n'
    #         '        normal on\n'
    #         '    }\n'
    #         '}\n')

    # ocean reflection on = 1, off = 0, original maps = 2
    if ocean_reflection == 0:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'diffuse', 0.85, 'brilliance', 1.0)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 diffuse 0.85 brilliance 1.0}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 1: #new version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'reflection', "{0.05,0.25}", 'diffuse', 0.85, 'brilliance', 1.0, 'phong', 0.06, 'phong_size', 30)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.03, 0.1>*0.6} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'

            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 2: #old version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(Pigment(ImageMap('jpeg', earth_month, "map_type", 1)), Finish(
        #                       'ambient', 0.0)),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           pigment{\n'
            '               image_map{\n'
            '                   jpeg ' + earth_month + ' map_type 1\n'
            '               }\n'
            '           }\n'
            '           finish{ambient 0.0}\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    # color_oceanblack: ocean black, continents white.
    # TextureMap 0.00: set color of the ocean.
    # TextureMap 1.00: use earth map for continents

    # atmos = Difference(Sphere([0, 0, 0], 3.504, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                    Sphere([0, 0, 0], 3.501, 'scale', [
    #                           1., R_Earth_pole/R_Earth, 1.]),
    #                    Material(Texture(Pigment("rgbt 1")), Interior(Media(Scattering(5, "color White", "eccentricity 0.56"),
    #                                                                        Density('spherical', DensityMap([0.0, "rgb 0.0"], [0.5294*0.25e-6, "rgb <0.02, 0.05, 0.2>*0.07"],
    #                                                                                                        [0.5294*0.4e-6, "rgb <0.02, 0.07, 0.3>*0.32"], [
    #                                                                            0.5294*0.5e-6,   "rgb <0.08, 0.18, 0.4>*0.5"],
    #                                                                            [0.5412*0.6e-6, "rgb <0.08, 0.18, 0.4>*0.9"], [
    #                                                                            0.5471*0.65e-6,  "rgb <0.08, 0.18, 0.4>*1.5"],
    #                                                                            [0.5471*0.675e-6, "rgb <0.08, 0.18, 0.4>*4.5"], [0.5471*0.71e-6,  "rgb <0.08, 0.18, 0.4>*12"]), "scale 3.52")))),
    #                    'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z')

    # pov_file.write(
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5003\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5001\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt 1\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     '                   scattering{\n'
    #     '                       4 color White\n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     '                           [0.0 rgb 0.0]\n'
    #     '                           [0.5294*0.25e-6 rgb <0.4, 0.5, 0.6>*0.07*10]\n'
    #     '                           [0.5294*0.4e-6 rgb <0.4, 0.5, 0.6>*0.32*10]\n'
    #     '                           [0.5294*0.5e-6 rgb <0.4, 0.5, 0.6>*0.5*10]\n'
    #     '                           [0.5412*0.6e-6 rgb <0.4, 0.5, 0.6>*0.9*10]\n'
    #     '                           [0.5471*0.65e-6 rgb <0.4, 0.5, 0.6>*1.5*10]\n'
    #     '                           [0.5471*0.685e-6 rgb <0.4, 0.5, 0.6>*4.5*10]\n'
    #     '                           [0.5471*0.69e-6 rgb <0.4, 0.6, 0.8>*4*10]\n'
    #     '                       }\n'
    #     '                       scale 3.5003\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 1\n'
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 0.989\n' #0.989
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt <1,1,1,1>\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
    #     '                   scattering { 4 rgb <0.4, 0.6, 0.8> \n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
    #     # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
    #     # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
    #     # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
    #     # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0 rgb <1,1,1>*0.]\n'
    #     # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
    #     # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
    #     # '                           [0.0039 rgb <1,1,1>*0.000]\n'
    #     '                           [0.0063 rgbt 0.0]\n'
    #     '                           [0.0085 rgbt 0.1]\n'
    #     # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
    #     '                           [0.0111 rgbt 1.0]\n'
    #     '                           [0.0111 rgbt 0.0]\n'
    #     '                           [1.0 rgbt 0.0]\n'
    #     '                       }\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
    #     '       scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'
    #     # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
    #     '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
    #     '#declare fBaseRadius = '+str(3.5)+';\n'
    #     '#declare fAtmoFactor = '+str(1.0114)+';\n'
    #     '#declare cAtmoColor = <1,1,1>;'
    #     '#declare M_Atmosphere_Intervals = 10;\n'
    #     '#declare M_Atmosphere_Samples = 10;\n'
    #     '#declare ATMO_STEPS = 9;\n'
    #     '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare M_Atmosphere =\n'
    #     '   media { \n'
    #     # '	    emission 0.3\n'
    #     '       scattering { 4 color rgb <1,1,1> \n'
    #     '	    intervals M_Atmosphere_Intervals\n'
    #     '	    samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
    #     '	    method 3\n'
    #     '	    density { spherical\n'
    #     '		    color_map {\n'
    #     '		        [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
    #     '               #local I=1;#while(I<ATMO_STEPS)\n'
    #     '               #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
    #     # '               #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
    #     '               #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
    #     # '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
    #     '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
    #     '               #local I=I+1;#end\n'
    #     # '			    [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '               [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
    #     '			    [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '			    [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
    #     '           }\n'
    #     '       }\n'
    #     '   }\n'
    #     '#declare atmos = \n'
    #     '   sphere { < 0, 0, 0>, 1\n'
    #     '       no_shadow\n'
    #     '       hollow off\n'        
    #     '       texture { pigment { P_EmittingAtmo } } // sun-side emission\n'        
    #     '       no_shadow\n'
    #     '       hollow on\n'        
    #     '       interior { media { M_Atmosphere } }\n'        
    #     '       finish { surface_finish }\n'
    #     '       scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    #https://github.com/pyramid3d/space3d/blob/master/include/s3d_scene.inc
    pov_file.write(
        # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
        '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
        '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
        '#declare fBaseRadius = '+str(3.5)+';\n'
        '#declare fAtmoFactor = '+str(1.0114)+';\n'
        '#declare cAtmoColor = <1,1,1>;'
        '#declare M_Atmosphere_Intervals = 1;\n'
        '#declare M_Atmosphere_Samples = 1;\n'
        '#declare ATMO_STEPS = 9;\n'
        '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_STRENGTH = 20;\n'
        '#declare ATMO_STRENGTH = 25;\n'
        # '#declare M_Atmosphere =\n'
        '#declare atmos = \n'
        # '   difference {\n'
        '   sphere {\n'
        '       <0, 0, 0>, 1\n'
        # '       scale <1., 1., 1.>\n'
        # '   }\n'
        # '       sphere {\n'
        # '           <0, 0, 0>, 0.989\n' #0.989
        # '           scale <1., 1., 1.>\n'
        # '       }\n'
        '       material{\n'
        '           texture{\n'
        '               pigment{\n'
        '                   rgbt <1,1,1,1>\n'
        '               }\n'
        '           }\n'
        '           interior{\n'
        '           media{\n'
    # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
        # '               scattering { 4 rgb <0.4, 0.6, 0.8> \n'
        '               scattering { 4 rgb <0.3, 0.4, 0.5> }\n'
        '	            intervals M_Atmosphere_Intervals\n'
        '	            samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
        '               density{\n'
        '                   spherical\n'
        '                   density_map {\n'
        '                       [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
        '                       #local I=1;#while(I<ATMO_STEPS)\n'
        '                       #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
        # '                           #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
        '                       #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
        # '                           [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
        '                       [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
        '                       #local I=I+1;#end\n'
        # '			                [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
        '                   }\n'
        # '                       density_map{\n'
        # # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
        # # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
        # # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
        # # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
        # # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0 rgb <1,1,1>*0.]\n'
        # # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
        # # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
        # # '                           [0.0039 rgb <1,1,1>*0.000]\n'
        # '                           [0.0063 rgbt 0.0]\n'
        # '                           [0.0085 rgbt 0.1]\n'
        # # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
        # '                           [0.0111 rgbt 1.0]\n'
        # '                           [0.0111 rgbt 0.0]\n'
        # '                           [1.0 rgbt 0.0]\n'
        # '                       }\n'
        '                       }\n'
        '                   }\n'
        '               }\n'
        '           }\n'
        # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
        # '           scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'        
        '           scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
        # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
        '           hollow on\n'        
        '           no_shadow\n'
        '           finish { surface_finish }\n'
        '           rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '           matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

        # '                           [1.0 rgb 0.]\n'
        # '                           [0.0109 rgb 0.]\n'
        # '                           [0.0109 rgb 1.]\n'
        # '                           [0.0103 rgb 0.5]\n'
        # '                           [0.0085 rgb 0.1]\n'
        # '                           [0.0063 rgb 0.01]\n'
        # '                           [0.0039 rgb 0.001]\n'
        # '                           [0.0009 rgb 0.0001]\n'
        # '                           [0.0 rgb 0.0]\n'
    
        # '                           [1.0 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.4, 0.6, 0.8>*10000]\n'
        # '                           [0.0103 rgb <0.4*0.5, 0.6*0.5, 0.8*0.5>*10000]\n'
        # '                           [0.0085 rgb <0.4*0.1, 0.6*0.1, 0.8*0.1>*10000]\n'
        # '                           [0.0063 rgb <0.4*0.01, 0.6*0.01, 0.8*0.01>*10000]\n'
        # '                           [0.0039 rgb <0.4*0.001, 0.6*0.001, 0.8*0.001>*10000]\n'
        # '                           [0.0009 rgb <0.4*0.0001, 0.6*0.0001, 0.8*0.0001>*10000]\n'
        # '                           [0.0 rgbt <0,0,0,1>*1000]\n'

    # clouds = Difference(Sphere([0, 0, 0], 3.502, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                     Sphere([0, 0, 0], 3.501, 'scale', [
    #                            1., R_Earth_pole/R_Earth, 1.]),
    #                     Texture(Pigment(ImagePattern('tiff', color_clouds_fileq, "map_type", 1),
    #                                     ColorMap([0.03, 'color', "rgbt <1,1,1,1>"], [1.0, 'color', "rgbt <1,1,1,0>"])),
    #                             Finish("ambient 0.0")), 'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                     'matrix', '<' + str(rotm_earth[0][0]) + ',' +
    #                     str(rotm_earth[0][1]) + ',' +
    #                     str(rotm_earth[0][2]) + ','
    #                     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #                     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>')
    # alternate cloud map
    #     clouds =    Difference(Sphere([0,0,0],3.502, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Sphere([0,0,0],3.501, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Texture(Pigment(ImagePattern('jpeg', cloud_fraction, "map_type", 1),
    #                 ColorMap([0.66, 'color', "rgbt <1,1,1,1>"],[0.95, 'color', "rgbt <1,1,1,0>"])),
    #                 Finish("ambient 0.0")),'hollow on','rotate','-90*x','scale','<1,1,-1>','rotate','180*z',
    #                 'matrix', '<' + str(rotm_earth[0][0]) + ',' + str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #                 + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    #                 + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')

    #0.03
    pov_file.write(
        '#declare clouds = \n'
        '   difference {\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.5004\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       sphere {\n'
        # '           <0, 0, 0>, 3.5001\n'
        '           <0, 0, 0>, 3.5002\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       texture {\n'
        '           pigment{\n'
        '               image_pattern{\n'
        '                   jpeg ' + color_clouds_fileq + ' map_type 1\n'
        '               }\n'
        '               color_map{\n'
        # '                   [0.15 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        # '                   [0.0 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'        
        '                   [0.0 color rgbt <0.8,0.8,0.8,1>]\n'
        '                   [0.3 color rgbt <0.7,0.7,0.7,0.99>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        '               }\n'
        '           }\n'
        '           finish{ambient 0.0}\n'
        '       }\n'
        '       hollow on\n'     
        '       no_shadow\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # ------------------------------------------------------------------------------------------------------
    # Moon Sphere
    # scaling factor for the Earth Moon environment (Earth primary central body)
    scale_m = R_Moon/R_Earth*3.5
    # imported values range from 0 to 1, transform to -1 to 1, then from -32768 to 32767, in this range, multiply by factor 0.5 to convert into meters
    # ele = 'ele=function{pigment{image_map{tiff ".Maps/moon/elevation_20.tiff" map_type 1 interpolate 2}}}'

 # Accuracy Parameter
    sc_dist = np.array(sc_pos) - np.array(moon_pos_e)
    dist = norm(sc_dist)    # in km
    if dist <= 3000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_1.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 10000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_2.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 20000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 30000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 70000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 100000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    else:
        acc = 0.001     
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
        
    pov_file.write(
    '#declare ele = \n'
    '   function{\n'
    '       pigment{\n'
    '           image_map{\n'
    '               tiff ' + elevation_fileq + ' map_type 1 interpolate 2\n'
    '           }\n'
    '       }\n'
    '   }\n')
    

#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture(Pigment(ImageMap('tiff', moon_darklight, "map_type", 1)),
#                 Finish('ambient', "0", 'diffuse', 0.7, 'brilliance', 1.5,)), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
        # The rotation of the sphere is done by using the matrix parameter (https://www.povray.org/documentation/view/3.6.0/49/) which takes directly the elements of the rotation matrix to compute the rotation. The rotation matrix is obtained through spice and it is from inertial (J2000) to body fixed (IAU_MOON)
        # adjust properties so we can use a single light source
#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture( ImagePattern( 'tiff', moon_craters2, "map_type", 1),
#                     TextureMap([0.000,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.8,'brilliance',1.2)],
#                                [0.001,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                [0.999,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                 [1.000, Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                                  Finish('ambient',0.0,'diffuse',0.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
    # sphere_m = Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
    #                       ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)
    #                                   ), 'accuracy', str(acc), 'max_gradient', 1.2,  # str(acc)
    #                       Texture(ImagePattern('tiff', moon_adj, "map_type", 1),
    #                               TextureMap([0.0, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.2, 'brilliance', 1.2)],
    #                                          [0.2, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.25)],
    #                                          [0.5, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.4)],
    #                                          [1.000, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.9, 'brilliance', 1.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                       'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
    #                        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
    #                                                          [1]) + ',' + str(rotm_moon[1][2]) + ','
    #                       + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
    #                                                          ) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
    #                       'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')

    pov_file.write(
        '#declare sphere_m = \n'
        '   isosurface {\n'
        '       function {\n'
        '           f_sphere(x,y,z,' + str(scale_m) +
        ') -  (ele(x,y,z).gray-0.5)*(2*32.767/2/' + str(R_Earth/3.5) + ')\n'
        '           }\n'
        '       contained_by {\n'
        '           sphere {0, (1 + ' + str(highest_pt /
                                            R_Moon) + ')*' + str(scale_m) + '}\n'
        '           }\n'
        '       accuracy ' + str(acc) + ' max_gradient 2.0\n'
        '       texture {\n'
        '           image_pattern{\n'
        '               tiff ' + moon_adj + ' map_type 1 interpolate 2\n'
        '           }\n'
        '           texture_map{\n'
        '               [0.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.2 brilliance 1.2}]\n'
        '               [0.2 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.25}]\n'
        '               [0.5 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.4}]\n'
        '               [1.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.9 brilliance 1.2}]\n'
        '           }\n'
        '       }\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_moon[0][0]) + ',' +
        str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
                                           [1]) + ',' + str(rotm_moon[1][2]) + ','
        + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
                                           ) + ',' + str(rotm_moon[2][2]) + ', 0, 0, 0>\n'
        '       translate <' + str(moon_pos_e[0]/R_Earth*3.5) + ',' + str(
            moon_pos_e[1]/R_Earth*3.5) + ',' + str(moon_pos_e[2]/R_Earth*3.5) + '>\n'
        '   }\n')

    # adjust regions -> 0,0.4,0.8,1.0
    # no need for moon_Craters2, just use moon_adj

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # scene = Scene(camera, objects=[light, sphere_e, clouds, sphere_m, atmos], included=[
    #     "functions.inc", "colors.inc"], declares=[ele])

    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    if look_at_object == 'EARTH':
        pov_file.write('object {sphere_e}\n')
        # pov_file.write('object {atmos}\n')
        pov_file.write('object {clouds}\n')
    elif look_at_object == 'MOON':
        pov_file.write('object {sphere_m}\n')

    if look_at_object == 'EARTH':
        print('EARTH IMAGE')
        if scenario==[]:
            image_name = "earth_img_" + str(int(iter)) + ".png"
        else:
            image_name = "earth_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            look_at = [0, 0, 0]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]
        
    elif look_at_object == 'MOON':
        print('MOON IMAGE')
        if scenario==[]:
            image_name = "moon_img_" + str(int(iter)) + ".png"
        else:
            image_name = "moon_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            # if dist-R_Moon<= 5000:
            #     sc_pos_rel_moon =  sc_pos-moon_pos_e
            #     norm_pos_rel_moon = np.linalg.norm(sc_pos_rel_moon)
            #     look_at = [moon_pos_e[0]/R_Earth*3.5 + sc_pos_rel_moon[0]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[1] /
            #             R_Earth*3.5 + sc_pos_rel_moon[1]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5 + sc_pos_rel_moon[2]/norm_pos_rel_moon*R_Moon/R_Earth*3.5]
            # else:
            look_at = [moon_pos_e[0]/R_Earth*3.5, moon_pos_e[1] /
                    R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]

    scene_file = os.path.join(os.getcwd(), image_name)
    scene_fileq = '"' + scene_file + '"'

    # camera = Camera('right', '<-' + str(cam_width/cam_height) + ',0,0>', 'sky', 'z', 'direction', 'z', 'angle', str(fov_angle),
    #                 'location', [sc_pos[0]/R_Earth*3.5, sc_pos[1] /
    #                              R_Earth*3.5, sc_pos[2]/R_Earth*3.5], 'up', 'z',
    #                 'look_at', look_at)

    # Camera with FOV and Resolution Implementation
    if len(sky_vec_input)==0:    
        sc2obj = np.array(look_at)-np.array(sc_pos)/R_Earth*3.5
        sc2obj = sc2obj/np.linalg.norm(sc2obj)
        # print(sc2obj)
        if norm(np.cross(sc2obj, np.array((0,0,1))))<1e-14:            
            x_b = np.array((1,0,0))           
            z_b = np.cross(x_b, sc2obj)
            sky_vec = '<' + str(z_b[0]) + ',' + str(z_b[1]) + ',' + str(z_b[2]) + '>'
            # print(sky_vec)
            # error('s')
        else:
            sky_vec = 'z'
        # Axis z the vertical of the camera plane
    else:
        sky_vec = '<' + str(sky_vec_input[0]) + ',' + str(sky_vec_input[1]) + ',' + str(sky_vec_input[2]) + '>'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    up z\n'
                   '    direction z\n'
                   '    sky ' + sky_vec + '\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' + str(sc_pos[0]/R_Earth*3.5) + ',' + str(
                       sc_pos[1]/R_Earth*3.5) + ',' + str(sc_pos[2]/R_Earth*3.5) + '>\n'
                   '    look_at <' +
                   str(look_at[0]) + ',' + str(look_at[1]) +
                   ',' + str(look_at[2]) + '>\n'
                   '}\n')

    # light source
    # light = LightSource([sun_pos_e[0]/R_Earth*3.5, sun_pos_e[1] /
    #                     R_Earth*3.5, sun_pos_e[2]/R_Earth*3.5], 'color', "rgb 2.4")

    pov_file.write('light_source { \n'
                   '    <' + str(sun_pos_e[0]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[1]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[2]/R_Earth*3.5) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    # width and height define the resolution of the image generated
    # scene.render(scene_file, width=cam_width, height=cam_height)

    ini_file = open(ini_namesingle,
                    'w')  # .ini temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        # 'Max_Image_Buffer_Memory=' + str(100000) + '\n'
        'Input_File_Name=' + '"' +
        os.path.join(os.getcwd(), pov_name) + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(os.getcwd(), ini_name) + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        print(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')
        os.system(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')

    stop = timeit.default_timer()
    print('End - Total Time: ' + str("{:.2f}".format(stop - start)) + ' s')

# OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(color_src)
    # plt.axis('off')
    # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()
# end render------------------------------------------------
    return scene_file, earth_map, cloud_map

def gen_moon_earth_combined(sc_eph, time_eph, camera_definition, iter, ocean_reflection, lookat_input=[], sky_vec_input=[], scenario=[], month_number=[], day=[], hour=[], testfolder=[], testtrain=[], testrange=[], testdate=[]):

    print('Start of image generation')
    start = timeit.default_timer()

    # determines month of the year -> changes earth map
    year = spice.timout(time_eph, 'YYYY')
    month = spice.timout(time_eph, 'MON')

    ### -------- INPUTS -------- ###
    # sc_eph:       spacecraft coordinates in J2000
    # time_eph:     ephemeris time
    # case:         Scenarios: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    # iter:         Iteration number for image generation purposes
    ### ------------------------ ###

    # Start of time count for the efficiency
    # print('Start of image generation')
    # start = timeit.default_timer()

    # Constants ------------------------------------------------------------------------------------------------------------------------------------------
    # km #scaling factor is: (variable in kilometers)/R_Earth*3.5
    R_Earth = 6378.1
    R_Earth_pole = 6356.8  # km
    highest_pt = 8.850  # km

    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    highest_pt = 11.000         # km

    # Calculations  ------------------------------------------------------------------------------------------------------------------------------------------
    # Position of the Moon compared to Earth in J2000 (ECI) reference frame
    moon = spice.spkpos("301", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    moon_pos_e = [moon[0][0], moon[0][1], moon[0][2]]

    # Position of the Spacecraft in the J2000 frame
    sc_pos = sc_eph

    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]

    # Position of Star Map: centered in Solar System Barycenter (spice ID: 0)
    star = spice.spkpos("0", time_eph, "J2000", "NONE", "399")
    star_pos_e = [star[0][0], star[0][1], star[0][2]]

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", time_eph)

    # Rotation of the Earth on its axis
    rotm_earth = spice.pxform("J2000", "IAU_EARTH", time_eph)

    # Visual Magnitude Parameters
    # Compute vector Earth to Sun
    earth2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    earth2sun = np.array([earth2sun[0][0], earth2sun[0][1], earth2sun[0][2]])
    # Compute the vector Moon to Sun
    moon2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "301")
    moon2sun = np.array([moon2sun[0][0], moon2sun[0][1], moon2sun[0][2]])
    # Compute the vector Earth to Spacecraft
    earth2sc = np.array(sc_pos)
    # Compute the vector Moon to Spacecraft
    moon2sc = np.array(sc_pos) - np.array(moon_pos_e)
    # Call function to calculate the Apparent Magnitude
    V_e, V_m, alpha_m = magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc)
    # print('The apparent magnitude of the Earth is: ', V_e)
    # print('The apparent magnitude of the Moon is: ', V_m)
    # print('The phase angle of the Moon is: '+str("{:.2f}".format(alpha_m))+' deg')

    #Cloud maps: https://www.shadedrelief.com/natural3/pages/clouds.html
    # Texture Maps Earth------------------------------------------------------------------------------------------------------------------------------------------
    if day.size==0:
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/earth/clouds.jpg") + '"'
        cloud_map = []
    else:       
        if month_number<10:
            str_month = "0" + str(int(month_number))
        else:
            str_month = str(int(month_number))
        if day<10:
            str_day = "0" + str(int(day))
        else:
            str_day = str(int(day))
        if hour<10:
            str_hour = "0" + str(int(hour))
        else:
            str_hour = str(int(hour))
        cloud_map = str(year) + "-" + str_month + "-" + str_day + "T" + str_hour + "-00-00.000Z.jpg"
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/clouds/allyear/" + cloud_map) + '"'
    # color_star_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/TychoSkymapII_v2.tiff") + '"'
    # new_clouds = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/pngkit-cloud-texture-png-961670.tif") + '"'
    # new_land = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_land.tiff") + '"'
    # color_oceanblack = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_oceanblackcontinentwhite.tiff") + '"'
    # color_Earth_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/world_shaded_scaled.jpg") + '"'
    # cloud_fraction = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/cloud_fraction.jpg") + '"'

    # Texture Maps Moon
    # color_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_color_scaled.tiff") + '"'
    # # elevation_fileq = '"' + \
    # #     os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'
    # new_moon = '"' + os.path.join(os.getcwd(), "Maps/moon/new_moon.tiff") + '"'
    # moon_craters2 = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_craters2.tiff") + '"'

    # maps for each month (WITHOUT ocean)
    earth_jan_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jan_adj.jpg") + '"'
    earth_feb_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_feb_adj.jpg") + '"'
    earth_mar_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_mar_adj.jpg") + '"'
    earth_apr_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_apr_adj.jpg") + '"'
    earth_may_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_may_adj.jpg") + '"'
    earth_jun_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jun_adj.jpg") + '"'
    earth_jul_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jul_adj.jpg") + '"'
    earth_aug_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_aug_adj.jpg") + '"'
    earth_sept_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept_adj.jpg") + '"'
    earth_oct_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_oct_adj.jpg") + '"'
    earth_nov_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_nov_adj.jpg") + '"'
    earth_dec_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_dec_adj.jpg") + '"'

    # maps for each month (WITH ocean)
    earth_jan = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jan.jpg") + '"'
    earth_feb = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_feb.jpg") + '"'
    earth_mar = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_mar.jpg") + '"'
    earth_apr = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_apr.jpg") + '"'
    earth_may = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_may.jpg") + '"'
    earth_jun = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jun.jpg") + '"'
    earth_jul = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jul.jpg") + '"'
    earth_aug = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_aug.jpg") + '"'
    earth_sept = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept.jpg") + '"'
    earth_oct = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_oct.jpg") + '"'
    earth_nov = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_nov.jpg") + '"'
    earth_dec = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_dec.jpg") + '"'

    # color_ocean_black_continent_white
    color_oceanblack_jan = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jan.jpg") + '"'
    color_oceanblack_feb = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_feb.jpg") + '"'
    color_oceanblack_mar = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_mar.jpg") + '"'
    color_oceanblack_apr = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_apr.jpg") + '"'
    color_oceanblack_may = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_may.jpg") + '"'
    color_oceanblack_jun = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jun.jpg") + '"'
    color_oceanblack_jul = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jul.jpg") + '"'
    color_oceanblack_aug = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_aug.jpg") + '"'
    color_oceanblack_sept = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_sept.jpg") + '"'
    color_oceanblack_oct = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_oct.jpg") + '"'
    color_oceanblack_nov = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_nov.jpg") + '"'
    color_oceanblack_dec = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_dec.jpg") + '"'

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------

    # determine the month
    if month == 'JAN':
        earth_month = earth_jan
        earth_month_adj = earth_jan_adj
        color_oceanblack_month = color_oceanblack_jan
        earth_map = "earth_jan.jpg"
        earth_adj = "earth_jan_adj.jpg"
        print('JAN')
    elif month == 'FEB':
        earth_month = earth_feb
        earth_month_adj = earth_feb_adj
        color_oceanblack_month = color_oceanblack_feb
        earth_map = "earth_feb.jpg"
        earth_adj = "earth_feb_adj.jpg"
        print('FEB')
    elif month == 'MAR':
        earth_month = earth_mar
        earth_month_adj = earth_mar_adj
        color_oceanblack_month = color_oceanblack_mar
        earth_map = "earth_mar.jpg"
        earth_adj = "earth_mar_adj.jpg"
        print('MAR')
    elif month == 'APR':
        earth_month = earth_apr
        earth_month_adj = earth_apr_adj
        color_oceanblack_month = color_oceanblack_apr
        earth_map = "earth_apr.jpg"
        earth_adj = "earth_apr_adj.jpg"
        print('APR')
    elif month == 'MAY':
        earth_month = earth_may
        earth_month_adj = earth_may_adj
        color_oceanblack_month = color_oceanblack_may
        earth_map = "earth_may.jpg"
        earth_adj = "earth_may_adj.jpg"
        print('MAY')
    elif month == 'JUN':
        earth_month = earth_jun
        earth_month_adj = earth_jun_adj
        color_oceanblack_month = color_oceanblack_jun
        earth_map = "earth_jun.jpg"
        earth_adj = "earth_jun_adj.jpg"
        print('JUN')
    elif month == 'JUL':
        earth_month = earth_jul
        earth_month_adj = earth_jul_adj
        color_oceanblack_month = color_oceanblack_jul
        earth_map = "earth_jul.jpg"
        earth_adj = "earth_jul_adj.jpg"
        print('JUL')
    elif month == 'AUG':
        earth_month = earth_aug
        earth_month_adj = earth_aug_adj
        color_oceanblack_month = color_oceanblack_aug
        earth_map = "earth_aug.jpg"
        earth_adj = "earth_aug_adj.jpg"
        print('AUG')
    elif month == 'SEP':
        earth_month = earth_sept
        earth_month_adj = earth_sept_adj
        color_oceanblack_month = color_oceanblack_sept
        earth_map = "earth_sept.jpg"
        earth_adj = "earth_sept_adj.jpg"
        print('SEP')
    elif month == 'OCT':
        earth_month = earth_oct
        earth_month_adj = earth_oct_adj
        color_oceanblack_month = color_oceanblack_oct
        earth_map = "earth_oct.jpg"
        earth_adj = "earth_oct_adj.jpg"
        print('OCT')
    elif month == 'NOV':
        earth_month = earth_nov
        earth_month_adj = earth_nov_adj
        color_oceanblack_month = color_oceanblack_nov
        earth_map = "earth_nov.jpg"
        earth_adj = "earth_nov_adj.jpg"
        print('NOV')
    elif month == 'DEC':
        earth_month = earth_dec
        earth_month_adj = earth_dec_adj
        color_oceanblack_month = color_oceanblack_dec
        earth_map = "earth_dec.jpg"
        earth_adj = "earth_dec_adj.jpg"
        print('DEC')

# Other Objects of Scene()
    if scenario==[]:
        pov_name = "tmp_" + str(int(iter)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '.ini'
    else:
        pov_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.ini'
    
    path_save = f'/home/gridsan/pmachuca/arclab_optnav_shared/yolov5/prefilter/{testfolder}/{testtrain}/{testrange}/{testdate}/images'
    pov_file = open(os.path.join(path_save, pov_namesingle),
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # pov_file.write('// GLOBAL SETTINGS\n'
    #         'global_settings { \n'
    #         '    assumed_gamma 1.0\n'
    #         '    radiosity { \n'
    #         '        count 1000\n'
    #         '        nearest_count 20\n'
    #         '        recursion_limit 1\n'
    #         '        normal on\n'
    #         '    }\n'
    #         '}\n')

    # ocean reflection on = 1, off = 0, original maps = 2
    if ocean_reflection == 0:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'diffuse', 0.85, 'brilliance', 1.0)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 diffuse 0.85 brilliance 1.0}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 1: #new version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'reflection', "{0.05,0.25}", 'diffuse', 0.85, 'brilliance', 1.0, 'phong', 0.06, 'phong_size', 30)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            #'                   [0.000 pigment{color rgb <0.007, 0.03, 0.1>*0.6} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'
            '                   [0.000 pigment{color rgb <0.007, 0.03, 0.1>*0.25} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'

            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 2: #old version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(Pigment(ImageMap('jpeg', earth_month, "map_type", 1)), Finish(
        #                       'ambient', 0.0)),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           pigment{\n'
            '               image_map{\n'
            '                   jpeg ' + earth_month + ' map_type 1\n'
            '               }\n'
            '           }\n'
            '           finish{ambient 0.0}\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    # color_oceanblack: ocean black, continents white.
    # TextureMap 0.00: set color of the ocean.
    # TextureMap 1.00: use earth map for continents

    # atmos = Difference(Sphere([0, 0, 0], 3.504, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                    Sphere([0, 0, 0], 3.501, 'scale', [
    #                           1., R_Earth_pole/R_Earth, 1.]),
    #                    Material(Texture(Pigment("rgbt 1")), Interior(Media(Scattering(5, "color White", "eccentricity 0.56"),
    #                                                                        Density('spherical', DensityMap([0.0, "rgb 0.0"], [0.5294*0.25e-6, "rgb <0.02, 0.05, 0.2>*0.07"],
    #                                                                                                        [0.5294*0.4e-6, "rgb <0.02, 0.07, 0.3>*0.32"], [
    #                                                                            0.5294*0.5e-6,   "rgb <0.08, 0.18, 0.4>*0.5"],
    #                                                                            [0.5412*0.6e-6, "rgb <0.08, 0.18, 0.4>*0.9"], [
    #                                                                            0.5471*0.65e-6,  "rgb <0.08, 0.18, 0.4>*1.5"],
    #                                                                            [0.5471*0.675e-6, "rgb <0.08, 0.18, 0.4>*4.5"], [0.5471*0.71e-6,  "rgb <0.08, 0.18, 0.4>*12"]), "scale 3.52")))),
    #                    'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z')

    # pov_file.write(
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5003\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5001\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt 1\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     '                   scattering{\n'
    #     '                       4 color White\n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     '                           [0.0 rgb 0.0]\n'
    #     '                           [0.5294*0.25e-6 rgb <0.4, 0.5, 0.6>*0.07*10]\n'
    #     '                           [0.5294*0.4e-6 rgb <0.4, 0.5, 0.6>*0.32*10]\n'
    #     '                           [0.5294*0.5e-6 rgb <0.4, 0.5, 0.6>*0.5*10]\n'
    #     '                           [0.5412*0.6e-6 rgb <0.4, 0.5, 0.6>*0.9*10]\n'
    #     '                           [0.5471*0.65e-6 rgb <0.4, 0.5, 0.6>*1.5*10]\n'
    #     '                           [0.5471*0.685e-6 rgb <0.4, 0.5, 0.6>*4.5*10]\n'
    #     '                           [0.5471*0.69e-6 rgb <0.4, 0.6, 0.8>*4*10]\n'
    #     '                       }\n'
    #     '                       scale 3.5003\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 1\n'
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 0.989\n' #0.989
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt <1,1,1,1>\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
    #     '                   scattering { 4 rgb <0.4, 0.6, 0.8> \n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
    #     # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
    #     # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
    #     # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
    #     # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0 rgb <1,1,1>*0.]\n'
    #     # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
    #     # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
    #     # '                           [0.0039 rgb <1,1,1>*0.000]\n'
    #     '                           [0.0063 rgbt 0.0]\n'
    #     '                           [0.0085 rgbt 0.1]\n'
    #     # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
    #     '                           [0.0111 rgbt 1.0]\n'
    #     '                           [0.0111 rgbt 0.0]\n'
    #     '                           [1.0 rgbt 0.0]\n'
    #     '                       }\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
    #     '       scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'
    #     # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
    #     '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
    #     '#declare fBaseRadius = '+str(3.5)+';\n'
    #     '#declare fAtmoFactor = '+str(1.0114)+';\n'
    #     '#declare cAtmoColor = <1,1,1>;'
    #     '#declare M_Atmosphere_Intervals = 10;\n'
    #     '#declare M_Atmosphere_Samples = 10;\n'
    #     '#declare ATMO_STEPS = 9;\n'
    #     '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare M_Atmosphere =\n'
    #     '   media { \n'
    #     # '	    emission 0.3\n'
    #     '       scattering { 4 color rgb <1,1,1> \n'
    #     '	    intervals M_Atmosphere_Intervals\n'
    #     '	    samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
    #     '	    method 3\n'
    #     '	    density { spherical\n'
    #     '		    color_map {\n'
    #     '		        [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
    #     '               #local I=1;#while(I<ATMO_STEPS)\n'
    #     '               #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
    #     # '               #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
    #     '               #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
    #     # '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
    #     '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
    #     '               #local I=I+1;#end\n'
    #     # '			    [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '               [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
    #     '			    [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '			    [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
    #     '           }\n'
    #     '       }\n'
    #     '   }\n'
    #     '#declare atmos = \n'
    #     '   sphere { < 0, 0, 0>, 1\n'
    #     '       no_shadow\n'
    #     '       hollow off\n'        
    #     '       texture { pigment { P_EmittingAtmo } } // sun-side emission\n'        
    #     '       no_shadow\n'
    #     '       hollow on\n'        
    #     '       interior { media { M_Atmosphere } }\n'        
    #     '       finish { surface_finish }\n'
    #     '       scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    #https://github.com/pyramid3d/space3d/blob/master/include/s3d_scene.inc
    pov_file.write(
        # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
        '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
        '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
        '#declare fBaseRadius = '+str(3.5)+';\n'
        '#declare fAtmoFactor = '+str(1.0114)+';\n'
        '#declare cAtmoColor = <1,1,1>;'
        '#declare M_Atmosphere_Intervals = 1;\n'
        '#declare M_Atmosphere_Samples = 1;\n'
        '#declare ATMO_STEPS = 9;\n'
        '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_STRENGTH = 20;\n'
        '#declare ATMO_STRENGTH = 25;\n'
        # '#declare M_Atmosphere =\n'
        '#declare atmos = \n'
        # '   difference {\n'
        '   sphere {\n'
        '       <0, 0, 0>, 1\n'
        # '       scale <1., 1., 1.>\n'
        # '   }\n'
        # '       sphere {\n'
        # '           <0, 0, 0>, 0.989\n' #0.989
        # '           scale <1., 1., 1.>\n'
        # '       }\n'
        '       material{\n'
        '           texture{\n'
        '               pigment{\n'
        '                   rgbt <1,1,1,1>\n'
        '               }\n'
        '           }\n'
        '           interior{\n'
        '           media{\n'
    # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
        # '               scattering { 4 rgb <0.4, 0.6, 0.8> \n'
        '               scattering { 4 rgb <0.3, 0.4, 0.5> }\n'
        '	            intervals M_Atmosphere_Intervals\n'
        '	            samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
        '               density{\n'
        '                   spherical\n'
        '                   density_map {\n'
        '                       [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
        '                       #local I=1;#while(I<ATMO_STEPS)\n'
        '                       #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
        # '                           #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
        '                       #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
        # '                           [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
        '                       [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
        '                       #local I=I+1;#end\n'
        # '			                [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
        '                   }\n'
        # '                       density_map{\n'
        # # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
        # # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
        # # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
        # # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
        # # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0 rgb <1,1,1>*0.]\n'
        # # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
        # # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
        # # '                           [0.0039 rgb <1,1,1>*0.000]\n'
        # '                           [0.0063 rgbt 0.0]\n'
        # '                           [0.0085 rgbt 0.1]\n'
        # # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
        # '                           [0.0111 rgbt 1.0]\n'
        # '                           [0.0111 rgbt 0.0]\n'
        # '                           [1.0 rgbt 0.0]\n'
        # '                       }\n'
        '                       }\n'
        '                   }\n'
        '               }\n'
        '           }\n'
        # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
        # '           scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'        
        '           scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
        # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
        '           hollow on\n'        
        '           no_shadow\n'
        '           finish { surface_finish }\n'
        '           rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '           matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

        # '                           [1.0 rgb 0.]\n'
        # '                           [0.0109 rgb 0.]\n'
        # '                           [0.0109 rgb 1.]\n'
        # '                           [0.0103 rgb 0.5]\n'
        # '                           [0.0085 rgb 0.1]\n'
        # '                           [0.0063 rgb 0.01]\n'
        # '                           [0.0039 rgb 0.001]\n'
        # '                           [0.0009 rgb 0.0001]\n'
        # '                           [0.0 rgb 0.0]\n'
    
        # '                           [1.0 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.4, 0.6, 0.8>*10000]\n'
        # '                           [0.0103 rgb <0.4*0.5, 0.6*0.5, 0.8*0.5>*10000]\n'
        # '                           [0.0085 rgb <0.4*0.1, 0.6*0.1, 0.8*0.1>*10000]\n'
        # '                           [0.0063 rgb <0.4*0.01, 0.6*0.01, 0.8*0.01>*10000]\n'
        # '                           [0.0039 rgb <0.4*0.001, 0.6*0.001, 0.8*0.001>*10000]\n'
        # '                           [0.0009 rgb <0.4*0.0001, 0.6*0.0001, 0.8*0.0001>*10000]\n'
        # '                           [0.0 rgbt <0,0,0,1>*1000]\n'

    # clouds = Difference(Sphere([0, 0, 0], 3.502, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                     Sphere([0, 0, 0], 3.501, 'scale', [
    #                            1., R_Earth_pole/R_Earth, 1.]),
    #                     Texture(Pigment(ImagePattern('tiff', color_clouds_fileq, "map_type", 1),
    #                                     ColorMap([0.03, 'color', "rgbt <1,1,1,1>"], [1.0, 'color', "rgbt <1,1,1,0>"])),
    #                             Finish("ambient 0.0")), 'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                     'matrix', '<' + str(rotm_earth[0][0]) + ',' +
    #                     str(rotm_earth[0][1]) + ',' +
    #                     str(rotm_earth[0][2]) + ','
    #                     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #                     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>')
    # alternate cloud map
    #     clouds =    Difference(Sphere([0,0,0],3.502, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Sphere([0,0,0],3.501, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Texture(Pigment(ImagePattern('jpeg', cloud_fraction, "map_type", 1),
    #                 ColorMap([0.66, 'color', "rgbt <1,1,1,1>"],[0.95, 'color', "rgbt <1,1,1,0>"])),
    #                 Finish("ambient 0.0")),'hollow on','rotate','-90*x','scale','<1,1,-1>','rotate','180*z',
    #                 'matrix', '<' + str(rotm_earth[0][0]) + ',' + str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #                 + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    #                 + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')

    #0.03
    pov_file.write(
        '#declare clouds = \n'
        '   difference {\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.5004\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       sphere {\n'
        # '           <0, 0, 0>, 3.5001\n'
        '           <0, 0, 0>, 3.5002\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       texture {\n'
        '           pigment{\n'
        '               image_pattern{\n'
        '                   jpeg ' + color_clouds_fileq + ' map_type 1\n'
        '               }\n'
        '               color_map{\n'
        # '                   [0.15 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        # '                   [0.0 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'        
        '                   [0.0 color rgbt <0.8,0.8,0.8,1>]\n'
        '                   [0.3 color rgbt <0.7,0.7,0.7,0.99>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        '               }\n'
        '           }\n'
        '           finish{ambient 0.0}\n'
        '       }\n'
        '       hollow on\n'     
        '       no_shadow\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # ------------------------------------------------------------------------------------------------------
    # Moon Sphere
    # scaling factor for the Earth Moon environment (Earth primary central body)
    scale_m = R_Moon/R_Earth*3.5
    # imported values range from 0 to 1, transform to -1 to 1, then from -32768 to 32767, in this range, multiply by factor 0.5 to convert into meters
    # ele = 'ele=function{pigment{image_map{tiff ".Maps/moon/elevation_20.tiff" map_type 1 interpolate 2}}}'

 # Accuracy Parameter
    sc_dist = np.array(sc_pos) - np.array(moon_pos_e)
    dist = norm(sc_dist)    # in km
    if dist <= 3000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_1.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 10000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_2.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 20000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 30000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 70000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 100000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    else:
        acc = 0.001     
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
        
    pov_file.write(
    '#declare ele = \n'
    '   function{\n'
    '       pigment{\n'
    '           image_map{\n'
    '               tiff ' + elevation_fileq + ' map_type 1 interpolate 2\n'
    '           }\n'
    '       }\n'
    '   }\n')
    

#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture(Pigment(ImageMap('tiff', moon_darklight, "map_type", 1)),
#                 Finish('ambient', "0", 'diffuse', 0.7, 'brilliance', 1.5,)), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
        # The rotation of the sphere is done by using the matrix parameter (https://www.povray.org/documentation/view/3.6.0/49/) which takes directly the elements of the rotation matrix to compute the rotation. The rotation matrix is obtained through spice and it is from inertial (J2000) to body fixed (IAU_MOON)
        # adjust properties so we can use a single light source
#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture( ImagePattern( 'tiff', moon_craters2, "map_type", 1),
#                     TextureMap([0.000,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.8,'brilliance',1.2)],
#                                [0.001,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                [0.999,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                 [1.000, Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                                  Finish('ambient',0.0,'diffuse',0.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
    # sphere_m = Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
    #                       ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)
    #                                   ), 'accuracy', str(acc), 'max_gradient', 1.2,  # str(acc)
    #                       Texture(ImagePattern('tiff', moon_adj, "map_type", 1),
    #                               TextureMap([0.0, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.2, 'brilliance', 1.2)],
    #                                          [0.2, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.25)],
    #                                          [0.5, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.4)],
    #                                          [1.000, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.9, 'brilliance', 1.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                       'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
    #                        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
    #                                                          [1]) + ',' + str(rotm_moon[1][2]) + ','
    #                       + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
    #                                                          ) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
    #                       'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')

    pov_file.write(
        '#declare sphere_m = \n'
        '   isosurface {\n'
        '       function {\n'
        '           f_sphere(x,y,z,' + str(scale_m) +
        ') -  (ele(x,y,z).gray-0.5)*(2*32.767/2/' + str(R_Earth/3.5) + ')\n'
        '           }\n'
        '       contained_by {\n'
        '           sphere {0, (1 + ' + str(highest_pt /
                                            R_Moon) + ')*' + str(scale_m) + '}\n'
        '           }\n'
        '       accuracy ' + str(acc) + ' max_gradient 2.0\n'
        '       texture {\n'
        '           image_pattern{\n'
        '               tiff ' + moon_adj + ' map_type 1 interpolate 2\n'
        '           }\n'
        '           texture_map{\n'
        '               [0.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.2 brilliance 1.2}]\n'
        '               [0.2 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.25}]\n'
        '               [0.5 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.4}]\n'
        '               [1.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.9 brilliance 1.2}]\n'
        '           }\n'
        '       }\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_moon[0][0]) + ',' +
        str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
                                           [1]) + ',' + str(rotm_moon[1][2]) + ','
        + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
                                           ) + ',' + str(rotm_moon[2][2]) + ', 0, 0, 0>\n'
        '       translate <' + str(moon_pos_e[0]/R_Earth*3.5) + ',' + str(
            moon_pos_e[1]/R_Earth*3.5) + ',' + str(moon_pos_e[2]/R_Earth*3.5) + '>\n'
        '   }\n')

    # adjust regions -> 0,0.4,0.8,1.0
    # no need for moon_Craters2, just use moon_adj

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # scene = Scene(camera, objects=[light, sphere_e, clouds, sphere_m, atmos], included=[
    #     "functions.inc", "colors.inc"], declares=[ele])

    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    if look_at_object == 'EARTH':
        pov_file.write('object {sphere_e}\n')
        pov_file.write('object {atmos}\n')
        pov_file.write('object {clouds}\n')
    elif look_at_object == 'MOON':
        pov_file.write('object {sphere_m}\n')

    if look_at_object == 'EARTH':
        print('EARTH IMAGE')
        if scenario==[]:
            image_name = "earth_img_" + str(int(iter)) + ".png"
        else:
            image_name = "earth_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            look_at = [0, 0, 0]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]
        
    elif look_at_object == 'MOON':
        print('MOON IMAGE')
        if scenario==[]:
            image_name = "moon_img_" + str(int(iter)) + ".png"
        else:
            image_name = "moon_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            # if dist-R_Moon<= 5000:
            #     sc_pos_rel_moon =  sc_pos-moon_pos_e
            #     norm_pos_rel_moon = np.linalg.norm(sc_pos_rel_moon)
            #     look_at = [moon_pos_e[0]/R_Earth*3.5 + sc_pos_rel_moon[0]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[1] /
            #             R_Earth*3.5 + sc_pos_rel_moon[1]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5 + sc_pos_rel_moon[2]/norm_pos_rel_moon*R_Moon/R_Earth*3.5]
            # else:
            look_at = [moon_pos_e[0]/R_Earth*3.5, moon_pos_e[1] /
                    R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]

    scene_file = os.path.join(path_save, image_name)
    scene_fileq = '"' + scene_file + '"'

    # camera = Camera('right', '<-' + str(cam_width/cam_height) + ',0,0>', 'sky', 'z', 'direction', 'z', 'angle', str(fov_angle),
    #                 'location', [sc_pos[0]/R_Earth*3.5, sc_pos[1] /
    #                              R_Earth*3.5, sc_pos[2]/R_Earth*3.5], 'up', 'z',
    #                 'look_at', look_at)

    # Camera with FOV and Resolution Implementation
    if len(sky_vec_input)==0:    
        sc2obj = np.array(look_at)-np.array(sc_pos)/R_Earth*3.5
        sc2obj = sc2obj/np.linalg.norm(sc2obj)
        # print(sc2obj)
        if norm(np.cross(sc2obj, np.array((0,0,1))))<1e-14:            
            x_b = np.array((1,0,0))           
            z_b = np.cross(x_b, sc2obj)
            sky_vec = '<' + str(z_b[0]) + ',' + str(z_b[1]) + ',' + str(z_b[2]) + '>'
            # print(sky_vec)
            # error('s')
        else:
            sky_vec = 'z'
        # Axis z the vertical of the camera plane
    else:
        sky_vec = '<' + str(sky_vec_input[0]) + ',' + str(sky_vec_input[1]) + ',' + str(sky_vec_input[2]) + '>'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    up z\n'
                   '    direction z\n'
                   '    sky ' + sky_vec + '\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' + str(sc_pos[0]/R_Earth*3.5) + ',' + str(
                       sc_pos[1]/R_Earth*3.5) + ',' + str(sc_pos[2]/R_Earth*3.5) + '>\n'
                   '    look_at <' +
                   str(look_at[0]) + ',' + str(look_at[1]) +
                   ',' + str(look_at[2]) + '>\n'
                   '}\n')

    # light source
    # light = LightSource([sun_pos_e[0]/R_Earth*3.5, sun_pos_e[1] /
    #                     R_Earth*3.5, sun_pos_e[2]/R_Earth*3.5], 'color', "rgb 2.4")

    pov_file.write('light_source { \n'
                   '    <' + str(sun_pos_e[0]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[1]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[2]/R_Earth*3.5) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    # width and height define the resolution of the image generated
    # scene.render(scene_file, width=cam_width, height=cam_height)

    # ini_file = open(ini_namesingle,
    #                 'w')  # .ini temporary file
    ini_file = open(os.path.join(path_save, ini_namesingle),
                    'w')  # scene temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        # 'Max_Image_Buffer_Memory=' + str(100000) + '\n'
        'Input_File_Name=' + '"' +
        os.path.join(path_save, pov_namesingle) + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(path_save, ini_namesingle) + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        print(povray_location + ' "' +
                  os.path.join(path_save, ini_namesingle) + '"')
        os.system(povray_location + ' "' +
                  os.path.join(path_save, ini_namesingle) + '"')

    stop = timeit.default_timer()
    print('End - Total Time: ' + str("{:.2f}".format(stop - start)) + ' s')

# OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(color_src)
    # plt.axis('off')
    # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()
# end render------------------------------------------------
    return scene_file, earth_map, cloud_map, 

def gen_moon_earth2(sc_eph, time_eph, camera_definition, iter, ocean_reflection, lookat_input=[], sky_vec_input=[]):

    print('Start of image generation')
    start = timeit.default_timer()

    # determines month of the year -> changes earth map
    month = spice.timout(time_eph, 'MON')

    ### -------- INPUTS -------- ###
    # sc_eph:       spacecraft coordinates in J2000
    # time_eph:     ephemeris time
    # case:         Scenarios: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    # iter:         Iteration number for image generation purposes
    ### ------------------------ ###

    # Start of time count for the efficiency
    # print('Start of image generation')
    # start = timeit.default_timer()

    # Constants ------------------------------------------------------------------------------------------------------------------------------------------
    # km #scaling factor is: (variable in kilometers)/R_Earth*3.5
    R_Earth = 6378.1
    R_Earth_pole = 6356.8  # km
    highest_pt = 8.850  # km

    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    highest_pt = 11.000         # km

    # Calculations  ------------------------------------------------------------------------------------------------------------------------------------------
    # Position of the Moon compared to Earth in J2000 (ECI) reference frame
    moon = spice.spkpos("301", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    moon_pos_e = [moon[0][0], moon[0][1], moon[0][2]]

    # Position of the Spacecraft in the J2000 frame
    sc_pos = sc_eph

    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]

    # Position of Star Map: centered in Solar System Barycenter (spice ID: 0)
    star = spice.spkpos("0", time_eph, "J2000", "NONE", "399")
    star_pos_e = [star[0][0], star[0][1], star[0][2]]

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", time_eph)

    # Rotation of the Earth on its axis
    rotm_earth = spice.pxform("J2000", "IAU_EARTH", time_eph)

    # Visual Magnitude Parameters
    # Compute vector Earth to Sun
    earth2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    earth2sun = np.array([earth2sun[0][0], earth2sun[0][1], earth2sun[0][2]])
    # Compute the vector Moon to Sun
    moon2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "301")
    moon2sun = np.array([moon2sun[0][0], moon2sun[0][1], moon2sun[0][2]])
    # Compute the vector Earth to Spacecraft
    earth2sc = np.array(sc_pos)
    # Compute the vector Moon to Spacecraft
    moon2sc = np.array(sc_pos) - np.array(moon_pos_e)
    # Call function to calculate the Apparent Magnitude
    V_e, V_m, alpha_m = magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc)
    # print('The apparent magnitude of the Earth is: ', V_e)
    # print('The apparent magnitude of the Moon is: ', V_m)
    # print('The phase angle of the Moon is: '+str("{:.2f}".format(alpha_m))+' deg')

    # Texture Maps Earth------------------------------------------------------------------------------------------------------------------------------------------
    color_clouds_fileq = '"' + \
        os.path.join(os.getcwd(), "Maps/earth/cloud_combined_2048.tif") + '"'
    color_star_fileq = '"' + \
        os.path.join(os.getcwd(), "Maps/earth/TychoSkymapII_v2.tiff") + '"'
    new_clouds = '"' + \
        os.path.join(
            os.getcwd(), "Maps/earth/pngkit-cloud-texture-png-961670.tif") + '"'
    new_land = '"' + \
        os.path.join(
            os.getcwd(), "Maps/earth/world_shaded_scaled_land.tiff") + '"'
    color_oceanblack = '"' + \
        os.path.join(
            os.getcwd(), "Maps/earth/world_shaded_scaled_oceanblackcontinentwhite.tiff") + '"'
    color_Earth_fileq = '"' + \
        os.path.join(os.getcwd(), "Maps/earth/world_shaded_scaled.jpg") + '"'
    cloud_fraction = '"' + \
        os.path.join(os.getcwd(), "Maps/earth/cloud_fraction.jpg") + '"'

    # Texture Maps Moon
    color_fileq = '"' + \
        os.path.join(os.getcwd(), "Maps/moon/moon_color_scaled.tiff") + '"'
    # elevation_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'
    new_moon = '"' + os.path.join(os.getcwd(), "Maps/moon/new_moon.tiff") + '"'
    moon_craters2 = '"' + \
        os.path.join(os.getcwd(), "Maps/moon/moon_craters2.tiff") + '"'

    # maps for each month (WITHOUT ocean)
    earth_jan_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jan_adj.jpg") + '"'
    earth_feb_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_feb_adj.jpg") + '"'
    earth_mar_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_mar_adj.jpg") + '"'
    earth_apr_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_apr_adj.jpg") + '"'
    earth_may_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_may_adj.jpg") + '"'
    earth_jun_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jun_adj.jpg") + '"'
    earth_jul_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jul_adj.jpg") + '"'
    earth_aug_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_aug_adj.jpg") + '"'
    earth_sept_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept_adj.jpg") + '"'
    earth_oct_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_oct_adj.jpg") + '"'
    earth_nov_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_nov_adj.jpg") + '"'
    earth_dec_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_dec_adj.jpg") + '"'

    # maps for each month (WITH ocean)
    earth_jan = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jan.jpg") + '"'
    earth_feb = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_feb.jpg") + '"'
    earth_mar = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_mar.jpg") + '"'
    earth_apr = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_apr.jpg") + '"'
    earth_may = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_may.jpg") + '"'
    earth_jun = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jun.jpg") + '"'
    earth_jul = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jul.jpg") + '"'
    earth_aug = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_aug.jpg") + '"'
    earth_sept = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept.jpg") + '"'
    earth_oct = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_oct.jpg") + '"'
    earth_nov = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_nov.jpg") + '"'
    earth_dec = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_dec.jpg") + '"'

    # color_ocean_black_continent_white
    color_oceanblack_jan = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jan.jpg") + '"'
    color_oceanblack_feb = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_feb.jpg") + '"'
    color_oceanblack_mar = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_mar.jpg") + '"'
    color_oceanblack_apr = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_apr.jpg") + '"'
    color_oceanblack_may = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_may.jpg") + '"'
    color_oceanblack_jun = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jun.jpg") + '"'
    color_oceanblack_jul = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jul.jpg") + '"'
    color_oceanblack_aug = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_aug.jpg") + '"'
    color_oceanblack_sept = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_sept.jpg") + '"'
    color_oceanblack_oct = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_oct.jpg") + '"'
    color_oceanblack_nov = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_nov.jpg") + '"'
    color_oceanblack_dec = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_dec.jpg") + '"'

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------

    # determine the month
    if month == 'JAN':
        earth_month = earth_jan
        earth_month_adj = earth_jan_adj
        color_oceanblack_month = color_oceanblack_jan
        print('JAN')
    elif month == 'FEB':
        earth_month = earth_feb
        earth_month_adj = earth_feb_adj
        color_oceanblack_month = color_oceanblack_feb
        print('FEB')
    elif month == 'MAR':
        earth_month = earth_mar
        earth_month_adj = earth_mar_adj
        color_oceanblack_month = color_oceanblack_mar
        print('MAR')
    elif month == 'APR':
        earth_month = earth_apr
        earth_month_adj = earth_apr_adj
        color_oceanblack_month = color_oceanblack_apr
        print('APR')
    elif month == 'MAY':
        earth_month = earth_may
        earth_month_adj = earth_may_adj
        color_oceanblack_month = color_oceanblack_may
        print('MAY')
    elif month == 'JUN':
        earth_month = earth_jun
        earth_month_adj = earth_jun_adj
        color_oceanblack_month = color_oceanblack_jun
        print('JUN')
    elif month == 'JUL':
        earth_month = earth_jul
        earth_month_adj = earth_jul_adj
        color_oceanblack_month = color_oceanblack_jul
        print('JUL')
    elif month == 'AUG':
        earth_month = earth_aug
        earth_month_adj = earth_aug_adj
        color_oceanblack_month = color_oceanblack_aug
        print('AUG')
    elif month == 'SEP':
        earth_month = earth_sept
        earth_month_adj = earth_sept_adj
        color_oceanblack_month = color_oceanblack_sept
        print('SEP')
    elif month == 'OCT':
        earth_month = earth_oct
        earth_month_adj = earth_oct_adj
        color_oceanblack_month = color_oceanblack_oct
        print('OCT')
    elif month == 'NOV':
        earth_month = earth_nov
        earth_month_adj = earth_nov_adj
        color_oceanblack_month = color_oceanblack_nov
        print('NOV')
    elif month == 'DEC':
        earth_month = earth_dec
        earth_month_adj = earth_dec_adj
        color_oceanblack_month = color_oceanblack_dec
        print('DEC')

# Other Objects of Scene()
    pov_file = open('tmp_' + str(int(iter)) + '.pov',
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # pov_file.write('// GLOBAL SETTINGS\n'
    #         'global_settings { \n'
    #         '    assumed_gamma 1.0\n'
    #         '    radiosity { \n'
    #         '        count 1000\n'
    #         '        nearest_count 20\n'
    #         '        recursion_limit 1\n'
    #         '        normal on\n'
    #         '    }\n'
    #         '}\n')

    # ocean reflection on = 1, off = 0, original maps = 2
    if ocean_reflection == 0:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'diffuse', 0.85, 'brilliance', 1.0)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 diffuse 0.85 brilliance 1.0}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 1:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'reflection', "{0.05,0.25}", 'diffuse', 0.85, 'brilliance', 1.0, 'phong', 0.06, 'phong_size', 30)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 2:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(Pigment(ImageMap('jpeg', earth_month, "map_type", 1)), Finish(
        #                       'ambient', 0.0)),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           pigment{\n'
            '               image_map{\n'
            '                   jpeg ' + earth_month + ' map_type 1\n'
            '               }\n'
            '           }\n'
            '           finish{ambient 0.0}\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    # color_oceanblack: ocean black, continents white.
    # TextureMap 0.00: set color of the ocean.
    # TextureMap 1.00: use earth map for continents

    # atmos = Difference(Sphere([0, 0, 0], 3.504, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                    Sphere([0, 0, 0], 3.501, 'scale', [
    #                           1., R_Earth_pole/R_Earth, 1.]),
    #                    Material(Texture(Pigment("rgbt 1")), Interior(Media(Scattering(5, "color White", "eccentricity 0.56"),
    #                                                                        Density('spherical', DensityMap([0.0, "rgb 0.0"], [0.5294*0.25e-6, "rgb <0.02, 0.05, 0.2>*0.07"],
    #                                                                                                        [0.5294*0.4e-6, "rgb <0.02, 0.07, 0.3>*0.32"], [
    #                                                                            0.5294*0.5e-6,   "rgb <0.08, 0.18, 0.4>*0.5"],
    #                                                                            [0.5412*0.6e-6, "rgb <0.08, 0.18, 0.4>*0.9"], [
    #                                                                            0.5471*0.65e-6,  "rgb <0.08, 0.18, 0.4>*1.5"],
    #                                                                            [0.5471*0.675e-6, "rgb <0.08, 0.18, 0.4>*4.5"], [0.5471*0.71e-6,  "rgb <0.08, 0.18, 0.4>*12"]), "scale 3.52")))),
    #                    'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z')

    pov_file.write(
        '#declare atmos = \n'
        '   difference {\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.504\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.501\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       material{\n'
        '           texture{\n'
        '               pigment{\n'
        '                   rgbt 1\n'
        '               }\n'
        '           }\n'
        '           interior{\n'
        '               media{\n'
        '                   scattering{\n'
        '                       5 color White eccentricity 0.56\n'
        '                   }\n'
        '                   density{\n'
        '                       spherical\n'
        '                       density_map{\n'
        '                           [0.0 rgb 0.0]\n'
        '                           [0.5294*0.25e-6 rgb <0.02, 0.05, 0.2>*0.07]\n'
        '                           [0.5294*0.4e-6 rgb <0.02, 0.07, 0.3>*0.32]\n'
        '                           [0.5294*0.5e-6 rgb <0.08, 0.18, 0.4>*0.5]\n'
        '                           [0.5412*0.6e-6 rgb <0.08, 0.18, 0.4>*0.9]\n'
        '                           [0.5471*0.65e-6 rgb <0.08, 0.18, 0.4>*1.5]\n'
        '                           [0.5471*0.675e-6 rgb <0.08, 0.18, 0.4>*4.5]\n'
        '                           [0.5471*0.71e-6 rgb <0.08, 0.18, 0.4>*12]\n'
        '                       }\n'
        '                       scale 3.504\n'
        '                   }\n'
        '               }\n'
        '           }\n'
        '       }\n'
        '       hollow on\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '   }\n')

    # clouds = Difference(Sphere([0, 0, 0], 3.502, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                     Sphere([0, 0, 0], 3.501, 'scale', [
    #                            1., R_Earth_pole/R_Earth, 1.]),
    #                     Texture(Pigment(ImagePattern('tiff', color_clouds_fileq, "map_type", 1),
    #                                     ColorMap([0.03, 'color', "rgbt <1,1,1,1>"], [1.0, 'color', "rgbt <1,1,1,0>"])),
    #                             Finish("ambient 0.0")), 'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                     'matrix', '<' + str(rotm_earth[0][0]) + ',' +
    #                     str(rotm_earth[0][1]) + ',' +
    #                     str(rotm_earth[0][2]) + ','
    #                     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #                     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>')
    # alternate cloud map
    #     clouds =    Difference(Sphere([0,0,0],3.502, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Sphere([0,0,0],3.501, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Texture(Pigment(ImagePattern('jpeg', cloud_fraction, "map_type", 1),
    #                 ColorMap([0.66, 'color', "rgbt <1,1,1,1>"],[0.95, 'color', "rgbt <1,1,1,0>"])),
    #                 Finish("ambient 0.0")),'hollow on','rotate','-90*x','scale','<1,1,-1>','rotate','180*z',
    #                 'matrix', '<' + str(rotm_earth[0][0]) + ',' + str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #                 + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    #                 + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')

    pov_file.write(
        '#declare clouds = \n'
        '   difference {\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.502\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.501\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       texture {\n'
        '           pigment{\n'
        '               image_pattern{\n'
        '                   tiff ' + color_clouds_fileq + ' map_type 1\n'
        '               }\n'
        '               color_map{\n'
        '                   [0.03 color rgbt <1,1,1,1>]\n'
        '                   [1.0 color rgbt <1,1,1,0>]\n'
        '               }\n'
        '           }\n'
        '           finish{ambient 0.0}\n'
        '       }\n'
        '       hollow on\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # ------------------------------------------------------------------------------------------------------
    # Moon Sphere
    # scaling factor for the Earth Moon environment (Earth primary central body)
    scale_m = R_Moon/R_Earth*3.5
    # imported values range from 0 to 1, transform to -1 to 1, then from -32768 to 32767, in this range, multiply by factor 0.5 to convert into meters
    # ele = 'ele=function{pigment{image_map{tiff ".Maps/moon/elevation_20.tiff" map_type 1 interpolate 2}}}'

 # Accuracy Parameter
    sc_dist = np.array(sc_pos) - np.array(moon_pos_e)
    dist = norm(sc_dist)    # in km
    if dist <= 3000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_1.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 10000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_2.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 20000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 30000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 70000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 100000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    else:
        acc = 0.001     
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
        
    pov_file.write(
    '#declare ele = \n'
    '   function{\n'
    '       pigment{\n'
    '           image_map{\n'
    '               tiff ' + elevation_fileq + ' map_type 1 interpolate 2\n'
    '           }\n'
    '       }\n'
    '   }\n')
    

#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture(Pigment(ImageMap('tiff', moon_darklight, "map_type", 1)),
#                 Finish('ambient', "0", 'diffuse', 0.7, 'brilliance', 1.5,)), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
        # The rotation of the sphere is done by using the matrix parameter (https://www.povray.org/documentation/view/3.6.0/49/) which takes directly the elements of the rotation matrix to compute the rotation. The rotation matrix is obtained through spice and it is from inertial (J2000) to body fixed (IAU_MOON)
        # adjust properties so we can use a single light source
#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture( ImagePattern( 'tiff', moon_craters2, "map_type", 1),
#                     TextureMap([0.000,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.8,'brilliance',1.2)],
#                                [0.001,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                [0.999,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                 [1.000, Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                                  Finish('ambient',0.0,'diffuse',0.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
    # sphere_m = Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
    #                       ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)
    #                                   ), 'accuracy', str(acc), 'max_gradient', 1.2,  # str(acc)
    #                       Texture(ImagePattern('tiff', moon_adj, "map_type", 1),
    #                               TextureMap([0.0, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.2, 'brilliance', 1.2)],
    #                                          [0.2, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.25)],
    #                                          [0.5, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.4)],
    #                                          [1.000, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.9, 'brilliance', 1.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                       'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
    #                        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
    #                                                          [1]) + ',' + str(rotm_moon[1][2]) + ','
    #                       + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
    #                                                          ) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
    #                       'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')

    pov_file.write(
        '#declare sphere_m = \n'
        '   isosurface {\n'
        '       function {\n'
        '           f_sphere(x,y,z,' + str(scale_m) +
        ') -  (ele(x,y,z).gray-0.5)*(2*32.767/2/' + str(R_Earth/3.5) + ')\n'
        '           }\n'
        '       contained_by {\n'
        '           sphere {0, (1 + ' + str(highest_pt /
                                            R_Moon) + ')*' + str(scale_m) + '}\n'
        '           }\n'
        '       accuracy ' + str(acc) + ' max_gradient 2.0\n'
        '       texture {\n'
        '           image_pattern{\n'
        '               tiff ' + moon_adj + ' map_type 1 interpolate 2\n'
        '           }\n'
        '           texture_map{\n'
        '               [0.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.2 brilliance 1.2}]\n'
        '               [0.2 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.25}]\n'
        '               [0.5 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.4}]\n'
        '               [1.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.9 brilliance 1.2}]\n'
        '           }\n'
        '       }\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_moon[0][0]) + ',' +
        str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
                                           [1]) + ',' + str(rotm_moon[1][2]) + ','
        + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
                                           ) + ',' + str(rotm_moon[2][2]) + ', 0, 0, 0>\n'
        '       translate <' + str(moon_pos_e[0]/R_Earth*3.5) + ',' + str(
            moon_pos_e[1]/R_Earth*3.5) + ',' + str(moon_pos_e[2]/R_Earth*3.5) + '>\n'
        '   }\n')

    # adjust regions -> 0,0.4,0.8,1.0
    # no need for moon_Craters2, just use moon_adj

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # scene = Scene(camera, objects=[light, sphere_e, clouds, sphere_m, atmos], included=[
    #     "functions.inc", "colors.inc"], declares=[ele])

    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    if look_at_object == 'EARTH':
        pov_file.write('object {sphere_e}\n')
        pov_file.write('object {atmos}\n')
        pov_file.write('object {clouds}\n')
    elif look_at_object == 'MOON':
        pov_file.write('object {sphere_m}\n')

    if look_at_object == 'EARTH':
        print('EARTH IMAGE')
        image_name = "earth_img_" + str(int(iter)) + ".png"
        if lookat_input==[]:
            look_at = [0, 0, 0]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]
        
    elif look_at_object == 'MOON':
        print('MOON IMAGE')
        image_name = "moon_img_" + str(int(iter)) + "_par.png"
        if lookat_input==[]:
            # if dist-R_Moon<= 5000:
            #     sc_pos_rel_moon =  sc_pos-moon_pos_e
            #     norm_pos_rel_moon = np.linalg.norm(sc_pos_rel_moon)
            #     look_at = [moon_pos_e[0]/R_Earth*3.5 + sc_pos_rel_moon[0]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[1] /
            #             R_Earth*3.5 + sc_pos_rel_moon[1]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5 + sc_pos_rel_moon[2]/norm_pos_rel_moon*R_Moon/R_Earth*3.5]
            # else:
            look_at = [moon_pos_e[0]/R_Earth*3.5, moon_pos_e[1] /
                    R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]

    scene_file = os.path.join(os.getcwd(), image_name)
    scene_fileq = '"' + scene_file + '"'

    # camera = Camera('right', '<-' + str(cam_width/cam_height) + ',0,0>', 'sky', 'z', 'direction', 'z', 'angle', str(fov_angle),
    #                 'location', [sc_pos[0]/R_Earth*3.5, sc_pos[1] /
    #                              R_Earth*3.5, sc_pos[2]/R_Earth*3.5], 'up', 'z',
    #                 'look_at', look_at)

    # Camera with FOV and Resolution Implementation
    if len(sky_vec_input)==0:
        sky_vec = 'z'
    else:
        sky_vec = '<' + str(sky_vec_input[0]) + ',' + str(sky_vec_input[1]) + ',' + str(sky_vec_input[2]) + '>'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    sky ' + sky_vec + '\n'
                   '    direction z\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' + str(sc_pos[0]/R_Earth*3.5) + ',' + str(
                       sc_pos[1]/R_Earth*3.5) + ',' + str(sc_pos[2]/R_Earth*3.5) + '>\n'
                   '    up z\n'
                   '    look_at <' +
                   str(look_at[0]) + ',' + str(look_at[1]) +
                   ',' + str(look_at[2]) + '>\n'
                   '}\n')

    # light source
    # light = LightSource([sun_pos_e[0]/R_Earth*3.5, sun_pos_e[1] /
    #                     R_Earth*3.5, sun_pos_e[2]/R_Earth*3.5], 'color', "rgb 2.4")

    pov_file.write('light_source { \n'
                   '    <' + str(sun_pos_e[0]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[1]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[2]/R_Earth*3.5) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    # width and height define the resolution of the image generated
    # scene.render(scene_file, width=cam_width, height=cam_height)

    ini_file = open('tmp_' + str(int(iter)) + '.ini',
                    'w')  # .ini temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        'Input_File_Name=' + '"' +
        os.path.join(os.getcwd(), "tmp_" +
                     str(int(iter)) + ".pov") + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(os.getcwd(), "tmp_" + str(int(iter)) + ".ini") + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        os.system(povray_location + ' "' +
                  os.path.join(os.getcwd(), "tmp_" + str(int(iter)) + ".ini") + '"')

    stop = timeit.default_timer()
    print('End - Total Time: ' + str("{:.2f}".format(stop - start)) + ' s')

# OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(color_src)
    # plt.axis('off')
    # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()
# end render------------------------------------------------
    return scene_file

# ---------------------### Conversion Rotation Matrix to Quaternions ###--------------------------

def extract_columns(file_path, delimiter=' '):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=delimiter)
        columns = zip(*reader)  # Transpose rows into columns

        extracted_columns = []
        for column in columns:
            extracted_columns.append([float(value) for value in column])

        return extracted_columns

def rpy_to_rot(roll, pitch, yaw):
    """
    Computes the rotation matrix from roll, pitch, and yaw angles (ZYX order).
    roll  = rotation about X-axis (ϕ)
    pitch = rotation about Y-axis (θ)
    yaw   = rotation about Z-axis (ψ)
    """

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])

    return R

def quat2rot(q1,q2,q3,q4):
    C11 = 1 - 2*q2**2 - 2*q3**2 
    C22 = 1 - 2*q1**2 - 2*q3**2 
    C33 = 1 - 2*q1**2 - 2*q2**2 

    C12 = 2*(q1*q2 - q3*q4) 
    C13 = 2*(q3*q1 + q2*q4) 
    C21 = 2*(q1*q2 + q3*q4) 
    C23 = 2*(q2*q3 - q1*q4) 
    C31 = 2*(q3*q1 - q2*q4) 
    C32 = 2*(q2*q3 + q1*q4) 

    C = np.array([
        [C11, C12, C13],
        [C21, C22, C23],
        [C31, C32, C33]
    ])
    
    return C

def rot2quat(C):
    # Computes quaternion elements from rotation matrix. Based on maximum element to avoid division by small number
    ### -------- INPUTS -------- ###
    # C = [3x3]                 rotation matrix
    ### -------- OUTPUTS ------- ###
    # q = [q1, q2, q3, q4]      4-element quaternion vector
    ### ------------------------ ###

    tr = C[0][0] + C[1][1] + C[2][2]

    if (tr > 0):
        S = math.sqrt(tr+1) * 2
        q4 = 0.25*S
        q1 = (C[2][1] - C[1][2]) / S
        q2 = (C[0][2] - C[2][0]) / S
        q3 = (C[1][0] - C[0][1]) / S
    elif (C[0][0] > C[1][1] & C[0][0] > C[2][2]):
        S = math.sqrt(1 + C[0][0] - C[1][1] - C[2][2]) * 2
        q4 = (C[2][1] - C[1][2]) / S
        q1 = 0.25 * S
        q2 = (C[0][1] + C[1][0]) / S
        q3 = (C[0][2] + C[2][0]) / S
    elif (C[1][1] > C[2][2]):
        S = math.sqrt(1 + C[1][1] - C[0][0] - C[2][2]) * 2
        q4 = (C[0][2] - C[2][0]) / S
        q1 = (C[0][1] + C[1][0]) / S
        q2 = 0.25 * S
        q3 = (C[1][2] + C[2][1]) / S
    else:
        S = math.sqrt(1 + C[2][2] - C[0][0] - C[1][1]) * 2
        q4 = (C[1][0] - C[0][1]) / S
        q1 = (C[0][2] + C[2][0]) / S
        q2 = (C[1][2] + C[2][1]) / S
        q3 = 0.25 * S

    q = [q1, q2, q3, q4]

    phi = 2*math.acos(q[3])
    axis = [q[0]/math.sin(phi/2), q[1]/math.sin(phi/2),
            q[2]/math.sin(phi/2), phi]
    return axis

# ---------------------### Apparent Magnitude Calculation ###--------------------------


def magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc):

    ### -------- INPUTS -------- ###
    # earth2sun:     Vector from Earth to Sun
    # moon2sun:      Vector from Moon to Sun
    # earth2sc:      Vector from Earth to S/C
    # moon2sc:       Vector from Moon to S/C
    ### -------- OUTPUTS ----------- ###
    # V_e:              Apparent Magnitude of Earth looking from S/C
    # V_m:              Apparent Magnitude of Moon looking from S/C
    # phase_angle_m:    Phase Angle of the Moon
    ### ------------------------ ###

    km2AU = 1/1.496E+8

    norms_e = (norm(earth2sun)*km2AU) * (norm(earth2sc)*km2AU)
    norms_m = (norm(moon2sun)*km2AU)*(norm(moon2sc)*km2AU)

    phase_angle_e = math.acos(
        np.dot(earth2sc, earth2sun)/(norm(earth2sc)*norm(earth2sun))) * 180/pi
    phase_angle_m = math.acos(
        np.dot(moon2sc, moon2sun)/(norm(moon2sc)*norm(moon2sun))) * 180/pi

    # Moon: This is inaccurate because it consides the model of Mercury and Moon to be similar, hence produces wrong values of apparent magnitude
    # V_m = 5*math.log10(norms_m) - 0.613 + 6.3280E-02*phase_angle_m - 1.6336E-03*phase_angle_m**2 \
    #     + 3.3644E-05*phase_angle_m**3 - 3.4265E-07*phase_angle_m**4 + 1.6893E-09*phase_angle_m**5 - 3.0334E-12*phase_angle_m**6

    # Moon
    V_m = 5*math.log10(norms_m) + 0.21 + 3.05*phase_angle_m/100 - \
        1.02*(phase_angle_m/100)**2 + 1.05*(phase_angle_m/100)**3

    # Earth
    V_e = 5*math.log10(norms_e) - 3.99 - 1.060E-3 * \
        phase_angle_e + 2.054E-4*phase_angle_e**2

    return V_e, V_m, phase_angle_m

# ---------------------### Exclusion of Earth or Moon from Scene ### -----------------------------


def exclusion(sc2moon, sc2earth, case):
    # Verifies the scenarios where the Moon or Earth are occultated
    ### -------- INPUTS -------- ###
    # sc2moon:          Vector from Spacecraft to Moon
    # sc2earth:         Vector from Spacecraft to Moon
    # case:             Choice of scenario: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    ### -------- OUTPUTS ----------- ###
    # moon:             Logical value for the exclusion of Moon (1 yes 0 no)
    # earth:            Logical value for the exclusion of Earth (1 yes 0 no)
    # angle_moon_earth: Angle between Earth and Moon from S/C POV
    ### ------------------------ ###

    angle_moon_earth = math.acos(
        np.dot(sc2moon, sc2earth)/(norm(sc2moon)*norm(sc2earth)))

    bEarth = norm(sc2earth)
    cEarth = 6378.1
    aEarth = math.sqrt(bEarth**2-cEarth**2)
    angle_horiz_Earth = math.acos(
        (aEarth**2+bEarth**2-cEarth**2)/(2*aEarth*bEarth))

    bMoon = norm(sc2moon)
    cMoon = 1737.4  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    aMoon = math.sqrt(bMoon**2-cMoon**2)
    angle_horiz_Moon = math.acos((aMoon**2+bMoon**2-cMoon**2)/(2*aMoon*bMoon))

    earth = 0
    moon = 0

    if case == 1:
        if angle_moon_earth <= angle_horiz_Moon and norm(sc2earth) > norm(sc2moon) or (angle_moon_earth + angle_horiz_Earth) <= angle_horiz_Moon:
            print('The Earth is behind the Moon')
            earth = 1
        else:
            earth = 0
    if case == 2 or case == 3:
        if angle_moon_earth <= angle_horiz_Earth and norm(sc2moon) > norm(sc2earth):
            print('The Moon is behind Earth')
            moon = 1
        else:
            moon = 0

    return moon, earth, angle_moon_earth

# ---------------------### Centroiding with Hough Circles Transform (Circle not Ellipse Fitting) ### -----------------------------


def hough_circles(img):
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    ### -------- OUTPUTS ----------- ###
    # plot:             Plot of Image with Center and Circle
    ### ------------------------ ###

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    cimg = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT,
                               1, 1500, param1=30, param2=15, minRadius=20, maxRadius=0)
    print('The center coordinates are: x = ' +
          str(circles[0][0][0]) + ' y = ' + str(circles[0][0][1]))
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 1, (0, 0, 255), 2)
    cv2.imwrite('centroid_houghcircles.png', cimg)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    plt.imshow(cimg)
    plt.show()

# ---------------------### Rotation Matrix from Body Frame to Inertial ### -----------------------------


def rot_body2inertial(r_sc, r_moon, rot_from_ME_to_J2000=[], lineofsight=[]):
    ### -------- INPUTS -------- ###
    # r_sc:             Position of the spacecraft in Inertial Frame
    # r_moon:           Position of the Moon in Inertial Frame
    ### -------- OUTPUTS ----------- ###
    # R_bi:             Rotation Matrix from Body Frame to Inertial
    ### ------------------------ ###

    if len(lineofsight)==0:
        r_sc = np.array(r_sc)
        r_moon = np.array(r_moon)

        # Axis y along the line of sight of the camera (assuming s/c pointed towards the center of the Moon and aligned camera shot)
        y_b = (r_moon - r_sc)/norm(r_moon - r_sc)
    else:
        y_b = lineofsight/norm(lineofsight)
    # Axis x the horizontal of the camera plane
    if len(rot_from_ME_to_J2000)==0:
        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])
    else:
        z_axis = np.array(rot_from_ME_to_J2000[:,2].flatten())
        x_axis = np.array(rot_from_ME_to_J2000[:,0].flatten())
    if norm(np.cross(y_b, z_axis))<1e-14:
        x_b = x_axis
    else:
        x_b = np.cross(y_b, z_axis) / \
            norm(np.cross(y_b, z_axis))
    # Axis z the vertical of the camera plane
    z_b = np.cross(x_b, y_b)

    R_bi = np.array([x_b, y_b, z_b]).T

    return R_bi

# ---------------------### Edge Detection with Canny Operator ### -----------------------------


def canny_edge(img, ksize):
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    # ksize:            Kernel Size for medianBlur (depending on distance from S/C)
    ### -------- OUTPUTS ----------- ###
    # edges:            Array with edges (image mask with edges)
    ### ------------------------ ###

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, ksize)
    # cv2.imwrite('earth_moon_img_blur.png', img_blur)
    edges = cv2.Canny(img_blur, 120, 245)

    return edges

# ---------------------### Edge Detection with Sobel Operator ### -----------------------------


def sobel_edge(img, ksize):
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    # ksize:            Kernel Size for medianBlur (depending on distance from S/C)
    ### -------- OUTPUTS ----------- ###
    # grad_x:           Array with horizontal edge gradient
    # grad_y:           Array with vertical edge gradient
    ### ------------------------ ###

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, ksize)
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    return grad_x, grad_y

# ---------------------### Canny Detection + Delete False Edges ### -----------------------------


def true_edge(img, R_ib, sun_vec_i, ksize):
    ### NOT USED ANYMORE ###
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    # R_ib:             Rotational Matrix from Inertial to Body Frame
    # sun_vec_i:        Sun Vector (from Sun to Moon) in Inertial Frame
    # ksize:            Kernel Size for medianBlur (depending on distance from S/C)
    ### -------- OUTPUTS ----------- ###
    # edges:            Array with edge points
    # edges_coord:      Matrix of edge in pixel coordinates (column vectors are coordinates)
    ### ------------------------ ###

    # Edge detection with Canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 122, 255, cv2.THRESH_BINARY)
    sun_vec_b = np.dot(R_ib, sun_vec_i)
    sun_vec_plane = np.hstack([sun_vec_b[0], -sun_vec_b[2]])
    rot_angle = pi - np.arctan2(sun_vec_plane[1], -sun_vec_plane[0])
    img_rot = ndimage.rotate(img_bin, rot_angle/pi*180, reshape=False)
    # cv2.imwrite('rot_bin.png', img_rot)
    kernel3 = np.array([[-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1],
                        [-1, -0.75, -0.5, -0.25, 0, 1, 1, 1, 1]])
    sharp_img = cv2.filter2D(src=img_rot, ddepth=-1, kernel=kernel3)
    # cv2.imwrite('test2.png', sharp_img)
    # canny1 = cv2.Canny(sharp_img, 127, 255)

    # cv2.imwrite('canny3.png', canny1)

    edges = canny_edge(img, 3)
    # cv2.imwrite('edges_canny.png', edges)

    # Calculate Edge Gradients with Sobel
    gX, gY = sobel_edge(img, 3)

    # Transform Sun Vector in Body Frame
    sun_vec_b = np.dot(R_ib, sun_vec_i)
    # Select the components on plane (z vertical and x horizontal in camera plane, the minus is to return to a conventional x direction)
    sun_vec_plane = np.hstack([sun_vec_b[0], -sun_vec_b[2]])

    rot_angle = np.arctan2(sun_vec_plane[1], sun_vec_plane[0])
    edges_rot = ndimage.rotate(edges, rot_angle/pi*180, reshape=False)

    # edges = rotate_edge(edges_rot, rot_angle)

    start_time = time.time()
    detect = 0
    for i in range(len(edges_rot)):
        for j in range(int(len(edges_rot[0])/2)):
            if edges_rot[i][j] > 30:
                detect = 0
                if edges_rot[i][j+1] > edges_rot[i][j]:
                    if edges_rot[i][j+2] > edges_rot[i][j+1]:
                        edges_rot[i][:] = 0
                        edges_rot[i][j+2] = 255
                    else:
                        edges_rot[i][:] = 0
                        edges_rot[i][j+1] = 255
                else:
                    edges_rot[i][:] = 0
                    edges_rot[i][j] = 255
                break
            else:
                detect += 1
        if detect != 0 and i > int(len(edges_rot)/2):
            break

    edges = ndimage.rotate(edges_rot, -rot_angle/pi*180, reshape=False)
    cv2.imwrite('canny_edge_rot.png', edges_rot)
    cv2.imwrite('canny_edge.png', edges)
    # print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    edges_coord = np.vstack((0, 0, 1.))
    # Check that edge gradient vector dot sun vector > 0 for removal of false edges (terminator)
    edge_count = -1
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            if edges[i][j] > 200:
                g = np.array([gX[i][j], gY[i][j]])
                prod = np.dot(g, sun_vec_plane)/(norm(g)*norm(sun_vec_plane))
                if (prod) < math.cos(65/180*pi):
                    edges[i][j] = 0
                else:
                    edge_count += 1
                    if edge_count == 0:
                        edges_coord = np.vstack((j+0.5, i+0.5, 1.))
                    else:
                        # +0.5 to use the correct pixel coordinates (in pixel frame)
                        edges_coord = np.hstack(
                            (edges_coord, np.vstack((j+0.5, i+0.5, 1.))))
            else:
                edges[i][j] = 0
    # print("--- %s seconds ---" % (time.time() - start_time))
    # cv2.imwrite('edges_canny_new.png', edges)

    return edges, edges_coord

# ---------------------### Improved Ellipse Fitting (Christian_JSR_2012) ### -----------------------------


def ellipse_fit(edge_coor):
    ### -------- INPUTS -------- ###
    # edge_coor:         Matrix of edge in pixel coordinates (column vectors are coordinates)
    ### -------- OUTPUTS ----------- ###
    # xc:               Pixel X of ellipse center
    # yc:               Pixel Y of ellipse center
    ### ------------------------ ###
    start_time = time.time()
    # aT*C*a = 1
    C1 = [[0, 0, 2],
          [0, -1, 0],
          [2, 0, 0]]

    # Matrix D for generalized eigenvalue problem (DT*D)a = lambda*C*a (improved to skip that problem)
    D1 = np.vstack((edge_coor[0, :]**2, edge_coor[0, :]
                   * edge_coor[1, :], edge_coor[1, :]**2)).T
    D2 = edge_coor.T

    # Scatter Matrix S = [S1 S2 ; S2T S3]
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)

    # Matrix M for simple eigenvalue problem M*a1 = lambda*a1
    C1inv, S3inv = np.linalg.inv(C1), np.linalg.inv(S3)
    M = np.dot(C1inv, S1 - np.dot(np.dot(S2, S3inv), S2.T))
    # Solve eigenvalue problem
    eigval, eigvec = np.linalg.eig(M)

    # Select the correct eigenvector (satisfy 4AC - B^2 > 0)
    flag = False
    for i in range(len(eigvec)):
        A, B, C = eigvec[0][i], eigvec[1][i], eigvec[2][i]
        if (4*A*C - B**2) > 0:
            a1 = np.array([A, B, C])
            flag = True
            break

    if flag == False:
        xc = 0
        yc = 0
        return xc, yc, flag
    else:

        # Solve a2 with a1
        a2 = np.dot(np.dot(-S3inv, S2.T), a1)

        # Implicit ellipse parameters
        A, B, C, D, E, F = a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]

        # Center coordinates calculation
        xc, yc = (2*C*D - B*E)/(B**2 - 4*A*C), (2*A*E - B*D)/(B**2 - 4*A*C)

        # print("Ellipse Fitting --- %s seconds ---" % (time.time() - start_time))

        # Semi-major and Semi-minor axes calculation
        smaj = np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + F*(B**2 - 4*A*C)) /
                       ((B**2 - 4*A*C)*(np.sqrt((A - C)**2 + B**2) - A - C)))
        smin = np.sqrt(2*(A*E**2 + C*D**2 - B*D*E + F*(B**2 - 4*A*C)) /
                       ((B**2 - 4*A*C)*(-np.sqrt((A - C)**2 + B**2) - A - C)))
        # Angle phi from x axis to semi-major axis (anti-clockwise)
        if B == 0:
            if A < C:
                phi = 0
            else:
                phi = pi/2
        elif B != 0:
            if A < C:
                phi = 0.5*(pi + np.arctan2(B, A-C))
            else:
                phi = pi/2 + 0.5*(np.arctan2(B, A-C))

        # Draw Ellipse in the original image
        # ellipse_fitted = Ellipse(
        #     (xc-0.5, yc-0.5), smaj*2, smin*2, (phi)*180/pi, edgecolor='g', fc='None', lw=2)

        # plt.figure()
        # plt.tight_layout()
        # # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ax = plt.gca()
        # ax.add_patch(ellipse_fitted)
        # color_plot = plt.imshow(img)
        # plt.savefig('ellipse_fit.pdf')
        # # plt.show()

        return xc, yc, flag

# ---------------------### Angular Accuracy of Centroiding Technique ### -----------------------------


def centroid_acc(xc, yc, camera_width, camera_height, foc_len_mm, px_mm):
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    # xc:               Pixel X of ellipse center
    # yc:               Pixel Y of ellipse center
    # foc_len:          Focal Length of the camera
    ### -------- OUTPUTS ----------- ###
    # xc:               Pixel X of ellipse center
    # yc:               Pixel Y of ellipse center
    ### ------------------------ ###

    xr, yr = camera_width/2, camera_height/2

    dx, dy = abs(xr - xc)*px_mm, abs(yr - yc)*px_mm
    d = math.sqrt(dx**2 + dy**2)

    err_angle_y, err_angle_x, err_angle_tot = math.atan2(
        dy, foc_len_mm), math.atan2(dx, foc_len_mm), math.atan2(d, foc_len_mm)

    return err_angle_y, err_angle_x, err_angle_tot

# ---------------------### Range Estimation from Centroiding ### -----------------------------


def pos_estimation(x_c, y_c, edge_coor, R_bi, camera_width, camera_height, focal_len_mm, px_mm):
    ### -------- INPUTS -------- ###
    # x_c:              Pixel X of ellipse center
    # y_c:              Pixel Y of ellipse center
    # edge_coor:        Matrix of edge in pixel coordinates (column vectors are coordinates)
    # edge_pts:         Array with edge points from edge detection
    # R_bi:             Rotation matrix from body to inertial frame
    ### -------- OUTPUTS ----------- ###
    # mu_rho:           Estimated Mean value of Range
    # sigma_rho:        Estimated Standard Deviation of Range
    ### ------------------------ ###

    start_time = time.time()

    # Calcuate Line of sight vector of center of Ellipse
    x_center_c, y_center_c, z_center_c = (-x_c + camera_width/2) * \
        px_mm, (-y_c + camera_height/2)*px_mm, focal_len_mm
    e_center_c = np.vstack([-x_center_c, -y_center_c, z_center_c])/norm(
        np.vstack([-x_center_c, -y_center_c, z_center_c]))    # In camera frame
    e_center_b = np.vstack(
        [-e_center_c[0], e_center_c[2], e_center_c[1]])   # In body frame
    e_center_i = np.dot(R_bi, e_center_b)   # In inertial frame

    # Principal Radii of Moon -> Body Matrix of Moon A_M
    # a, b, c = 1739.088, 1737.37, 1734.969    # km
    a = 1737.8981  # km, should be 1737.8981, from "A New Global Database of Lunar Impact Craters"
    b = 1737.8981  # km, should be 1737.8981
    c = 1735.6576  # km
    A_M = [[1/a**2, 0, 0],
           [0, 1/b**2, 0],
           [0, 0, 1/c**2]]

    # Calculate Line of sight vector of ellipse points + Estimate Range
    rho_vec = np.zeros(len(edge_coor[0]))
    lengthx = camera_width
    lengthy = camera_height
    for i in range(len(edge_coor[0])):
        # Extract ellipse coordinates in Image Frame
        x_im, y_im = edge_coor[0][i], edge_coor[1][i]
        # Convert in Camera Frame (mm units)
        x_c, y_c = (x_im - lengthx/2)*px_mm, (-y_im + lengthy/2)*px_mm
        # Line of Sight of Ellipse Points in Camera Frame
        z_c = focal_len_mm    # mm
        e_c = np.vstack([-x_c, -y_c, z_c])/norm(np.vstack([-x_c, -y_c, z_c]))
        # In Body Frame
        e_b = np.vstack([-e_c[0], e_c[2], e_c[1]])
        # Convert in Inertial Frame
        e_i = np.dot(R_bi, e_b)
        # Range Estimation
        rho = 1/np.sqrt(np.dot(np.dot(e_center_i.T, A_M), e_center_i) - (
            (np.dot(np.dot(e_center_i.T, A_M), e_i))**2)/(np.dot(np.dot(e_i.T, A_M), e_i)))
        if np.iscomplexobj(rho):
            # print(rho)
            rho = rho.real
        rho_vec[i] = rho

    # Mean value and Standard Deviation
    mu_rho, sigma_rho = np.mean(rho_vec), np.std(rho_vec)

    # print('Mean Value of Range: ' + str("{:.3f}".format(mu_rho)) + ' km, Standard Deviation of Range: ' + str("{:.3f}".format(sigma_rho)) + ' km' )
    # print("Pose Estimation --- %s seconds ---" % (time.time() - start_time))
    return mu_rho, sigma_rho

# ---------------------### Rotation Matrix from Camera Frame to MOON_ME ### -----------------------------


def rot_camera2me(r_sc, r_moon):
    ### -------- INPUTS -------- ###
    # r_sc:             Position of the spacecraft in Inertial Frame
    # r_moon:           Position of the Moon in Inertial Frame
    ### -------- OUTPUTS ----------- ###
    # R_ci:             Rotation Matrix from Camera Frame to ME
    ### ------------------------ ###

    r_sc = np.hstack(r_sc)
    r_moon = np.hstack(r_moon)

    # Axis z along the line of sight of the camera (assuming s/c pointed towards the center of the Moon and aligned camera shot)
    z_c = (r_moon - r_sc)/norm(r_moon - r_sc)
    # Axis x the horizontal of the camera plane
    x_c = np.cross(z_c, np.array([0, 0, 1])) / \
        norm(np.cross(z_c, np.array([0, 0, 1])))
    # Axis y the vertical of the camera plane
    y_c = np.cross(z_c, x_c)

    R_ci = np.array([x_c, y_c, z_c]).T

    return R_ci

# ---------------------### Rotation Matrix from Camera Frame to MOON_PA ### -----------------------------


def rot_camera2pa(r_sc, r_moon):
    ### -------- INPUTS -------- ###
    # r_sc:             Position of the spacecraft in Inertial Frame
    # r_moon:           Position of the Moon in Inertial Frame
    ### -------- OUTPUTS ----------- ###
    # R_cpa:             Rotation Matrix from Camera Frame to PA
    ### ------------------------ ###

    r_sc = np.hstack(r_sc)
    r_moon = np.hstack(r_moon)

    # Axis y along the line of sight of the camera (assuming s/c pointed towards the center of the Moon and aligned camera shot)
    z_c = (r_moon - r_sc)/norm(r_moon - r_sc)
    # Axis x the horizontal of the camera plane
    x_c = np.cross(z_c, np.array([0, 0, 1])) / \
        norm(np.cross(z_c, np.array([0, 0, 1])))
    # Axis z the vertical of the camera plane
    y_c = np.cross(z_c, x_c)

    R_ci = np.array([x_c, y_c, z_c]).T

    return R_ci

# ---------------------### Projection of 3D Ellipsoid onto plane for Perspective Projection Offset ### -----------------------------


def ellipsoid_projection(sc_pos, moon_pos, date, camera_definition, *scene_file):
    ### -------- INPUTS -------- ###
    # sc_pos:             Position of the spacecraft in Inertial Frame
    # moon_pos:           Position of the Moon in Inertial Frame
    # date:               Ephemeris Time
    ### -------- OUTPUTS ----------- ###
    # x_off:              x Offset of Projected Ellipse Center
    # y_off:              y Offset of Projected Ellipse Center
    # moon_area_pixels:   area of the moon in pixels
    ### ------------------------ ###

    start_time = time.time()

    sc_pos, moon_pos = np.vstack(sc_pos), np.vstack(moon_pos)
    # 3D Ellipsoid projection (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7035841)
    # Euler angles from SELENE Sampled
    alpha, beta, gamma = 17.39/180*pi, 21.24/180*pi, 27.33/180*pi
    # Rotation matrix to have geometric center of Moon from ME (R_geom is from ME to PA)
    R_geom1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [
                       0, -np.sin(alpha), np.cos(alpha)]])
    R_geom2 = np.array([[np.cos(beta), 0, -np.sin(beta)],
                       [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    R_geom3 = np.array([[np.cos(gamma), np.sin(gamma), 0],
                       [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R_geom = np.matmul(np.matmul(R_geom1, R_geom2), R_geom3)  # me2PA
    # Radii of the Moon (SELENE Sampled)
    # a, b, c = 1739.088, 1737.37, 1734.969    # km
    a = 1737.1513  # km, from "A New Global Database of Lunar Impact Craters"
    b = 1737.1513  # km
    c = 1735.6576  # km
    # Covariance Matrix
    A_Moon = np.array([[a**2, 0, 0], [0, b**2, 0], [0, 0, c**2]])
    # Rotation matrix from Inertial Frame to ME
    R_i2ME = spice.pxform("J2000", "MOON_ME", date)
    # Transform sc_pos and moon_pos in ME
    sc_pos_me, moon_pos_me = np.dot(R_i2ME, sc_pos), np.dot(R_i2ME, moon_pos)
    # Add the offset of the center
    r_offset = np.vstack((-1.736, -.723, .226))    # Geometric center offset
    moon_pos_me_geom = moon_pos_me + r_offset
    # Compute the rotation matrix from ME to camera frame
    R_c2me = rot_camera2me(sc_pos_me, moon_pos_me)
    R_me2c = R_c2me.T
    # Covariance Matrix in Camera Frame
    A = np.dot(np.dot(np.dot(np.dot(R_c2me.T, R_geom.T), A_Moon), R_geom), R_c2me)
    geom_center_c = np.dot(R_me2c, moon_pos_me_geom - sc_pos_me)
    Ainv = np.linalg.inv(A)
    M = np.dot(np.dot(Ainv, geom_center_c), np.dot(geom_center_c.T, Ainv.T)) - \
        (np.dot(np.dot(geom_center_c.T, Ainv), geom_center_c)-1)*Ainv
    # Camera Calibration Matrix
    fx, fy = 35/4.96E-3, 35/4.96E-3
    K = np.vstack((np.hstack((fx, 0, 0)), np.hstack(
        (0, fy, 0)), np.hstack((0, 0, 1))))
    # Extraction of Sub-Matrix
    M31, M23, M33, K33 = M[0:2, 1:3], M[(0, 2), 0:2], M[0:2, 0:2], K[0:2, 0:2]
    # Solve mu_p = (x_off, y_off)
    mu_p = 1/np.linalg.det(M33)*np.dot(K33,
                                       np.vstack((np.linalg.det(M31), -np.linalg.det(M23))))
    ep_p = -np.linalg.det(M)/np.linalg.det(M33) * \
        np.dot(np.dot(K33, np.linalg.inv(M33)), K33.T)

    x_off, y_off = mu_p[0], mu_p[1]

    # print("Ellipsoid Projection --- %s seconds ---" % (time.time() - start_time))
    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    # Convert types after unpacking
    focal_len = int(focal_len)
    cam_width = int(cam_width)
    cam_height = int(cam_height)
    px = float(px)
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    # # OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(1, figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)
    # color_plot = plt.imshow(color_src)

    eigval, eigvec = np.linalg.eig(ep_p)

    # plt.plot(cam_width/2+mu_p[0]-0.5, cam_height/2+mu_p[1]-0.5, 'rx')

    theta = np.deg2rad(np.arange(0.0, 360.0, 1.))
    x = np.sqrt(eigval[0]) * np.cos(theta)
    y = np.sqrt(eigval[1]) * np.sin(theta)
    moon_area_pixels = np.pi*np.sqrt(eigval[0])*np.sqrt(eigval[1])
    # R = eigvec
    # x, y = np.dot(R, np.array([x, y]))
    # x += cam_width/2+mu_p[0]
    # y += cam_height/2+mu_p[1]
    # plt.plot(x-0.5, y-0.5, 'r')
    # plt.show()

    return x_off, y_off, moon_area_pixels

def ellipsoid_projection_earth(sc_pos, earth_pos, date, camera_definition, *scene_file):
    ### -------- INPUTS -------- ###
    # sc_pos:             Position of the spacecraft in Inertial Frame
    # earth_pos:          Position of the Earth in Inertial Frame
    # date:               Ephemeris Time
    ### -------- OUTPUTS ----------- ###
    # x_off:              x Offset of Projected Ellipse Center
    # y_off:              y Offset of Projected Ellipse Center
    # earth_area_pixels:   area of the earth in pixels
    ### ------------------------ ###

    start_time = time.time()

    sc_pos, earth_pos = np.vstack(sc_pos), np.vstack(earth_pos)

    # Axes of Earth
    a = 6378.17136  # km, from https://link.springer.com/article/10.1007/s00190-023-01717-1
    b = 6378.10161  # km
    c = 6356.75183  # km
    # Covariance Matrix
    A_Earth = np.array([[a**2, 0, 0], [0, b**2, 0], [0, 0, c**2]])
    # Rotation matrix from Inertial Frame to IAU
    R_i2ME = spice.pxform("J2000", "IAU_EARTH", date)
    # Transform sc_pos and earth_pos in IAU
    sc_pos_me, earth_pos_me = np.dot(R_i2ME, sc_pos), np.dot(R_i2ME, earth_pos)
    earth_pos_me_geom = earth_pos_me
    # Compute the rotation matrix from IAU to camera frame
    R_c2me = rot_camera2me(sc_pos_me, earth_pos_me)
    R_me2c = R_c2me.T
    # Covariance Matrix in Camera Frame
    A = np.dot(np.dot(R_c2me.T, A_Earth), R_c2me)
    geom_center_c = np.dot(R_me2c, earth_pos_me_geom - sc_pos_me)
    Ainv = np.linalg.inv(A)
    M = np.dot(np.dot(Ainv, geom_center_c), np.dot(geom_center_c.T, Ainv.T)) - \
        (np.dot(np.dot(geom_center_c.T, Ainv), geom_center_c)-1)*Ainv
    # Camera Calibration Matrix
    fx, fy = 35/4.96E-3, 35/4.96E-3
    K = np.vstack((np.hstack((fx, 0, 0)), np.hstack(
        (0, fy, 0)), np.hstack((0, 0, 1))))
    # Extraction of Sub-Matrix
    M31, M23, M33, K33 = M[0:2, 1:3], M[(0, 2), 0:2], M[0:2, 0:2], K[0:2, 0:2]
    # Solve mu_p = (x_off, y_off)
    mu_p = 1/np.linalg.det(M33)*np.dot(K33,
                                       np.vstack((np.linalg.det(M31), -np.linalg.det(M23))))
    ep_p = -np.linalg.det(M)/np.linalg.det(M33) * \
        np.dot(np.dot(K33, np.linalg.inv(M33)), K33.T)

    x_off, y_off = mu_p[0], mu_p[1]

    # print("Ellipsoid Projection --- %s seconds ---" % (time.time() - start_time))
    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    # Convert types after unpacking
    focal_len = int(focal_len)
    cam_width = int(cam_width)
    cam_height = int(cam_height)
    px = float(px)
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    # # OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(1, figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)
    # color_plot = plt.imshow(color_src)

    eigval, eigvec = np.linalg.eig(ep_p)

    # plt.plot(cam_width/2+mu_p[0]-0.5, cam_height/2+mu_p[1]-0.5, 'rx')

    theta = np.deg2rad(np.arange(0.0, 360.0, 1.))
    x = np.sqrt(eigval[0]) * np.cos(theta)
    y = np.sqrt(eigval[1]) * np.sin(theta)
    earth_area_pixels = np.pi*np.sqrt(eigval[0])*np.sqrt(eigval[1])
    # R = eigvec
    # x, y = np.dot(R, np.array([x, y]))
    # x += cam_width/2+mu_p[0]
    # y += cam_height/2+mu_p[1]
    # plt.plot(x-0.5, y-0.5, 'r')
    # plt.show()

    return x_off, y_off, earth_area_pixels

# ---------------------### Christian-Robinson Algorithm for Relative Position Estimation ### -----------------------------


def christian_robinson(sc_pos, moon_pos, edge_coor, date, camera_width, camera_height, focal_len_mm, px_mm):
    ### -------- INPUTS -------- ###
    # sc_pos:             Position of the spacecraft in Inertial Frame
    # moon_pos:           Position of the Moon in Inertial Frame
    # date:               Ephemeris Time
    ### -------- OUTPUTS ----------- ###
    # x_off:              x Offset of Projected Ellipse Center
    # y_off:              y Offset of Projected Ellipse Center
    ### ------------------------ ###

    start_time = time.time()
    # Compute D = diag[1/a, 1/b, 1/c]
    # a = 1739.088    # km
    # b = 1737.37     # km
    # c = 1734.969    # km
    a = 1737.8981  # km, should be 1737.8981, from "A New Global Database of Lunar Impact Craters"
    b = 1737.8981  # km, should be 1737.8981
    c = 1735.6576  # km
    A_m = np.array([[1/a**2, 0, 0],
                    [0, 1/b**2, 0],
                    [0, 0, 1/c**2]])
    # Euler angles from SELENE Sampled
    alpha = 17.39/180*pi
    beta = 21.24/180*pi
    gamma = 27.33/180*pi
    # Rotation matrix from ME to PA
    R_geom1 = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [
                       0, -np.sin(alpha), np.cos(alpha)]])
    R_geom2 = np.array([[np.cos(beta), 0, -np.sin(beta)],
                       [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    R_geom3 = np.array([[np.cos(gamma), np.sin(gamma), 0],
                       [-np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R_geom = np.matmul(np.matmul(R_geom1, R_geom2), R_geom3)  # ME to PA
    # Simple A_m decomposition A_m = D.T * D
    D = np.array([[1/a, 0, 0],
                  [0, 1/b, 0],
                  [0, 0, 1/c]])
    # Compute the rotation matrix from Inertial J2000 to ME
    R_i2ME = spice.pxform("J2000", "MOON_ME", date)
    # Convert position vectors in ME Frame
    sc_pos_me = np.dot(R_i2ME, sc_pos)
    moon_pos_me = np.dot(R_i2ME, moon_pos)
    # Compute Rotation Matrix from Camera Frame (z-axis along line of sight, x-axis to the right looking from front, y-axis to bottom) to ME
    R_c2ME = rot_camera2me(sc_pos_me, moon_pos_me)
    R_ME2c = R_c2ME.T
    # Compute Rotation Matrix from Camera Frame to PA
    R_c2PA = np.dot(R_geom, R_c2ME)
    # Camera Calibration matrix K
    mu_x = px_mm  # mm (from center pixel to center pixel)
    mu_y = px_mm
    dx = focal_len_mm/mu_x
    dy = focal_len_mm/mu_y
    u_p = camera_width/2  # coord of center in image frame
    v_p = camera_height/2   # coord of center in image frame
    K_inv = np.array([[1/dx,   0,      -dy*u_p/(dx*dy)],
                      [0,     1/dy,       -v_p/dy],
                      [0,      0,              1]])
    # Direct Rotation Matrix
    R = np.dot(np.dot(D, R_c2PA), K_inv)
    # Populate array H (edge_coor is a matrix with column vectors [x, y, 1] of edge points)
    H = np.zeros((3, np.size(edge_coor, 1)))
    for i in range(np.size(edge_coor, 1)):
        x_dash = np.dot(R, edge_coor[:, [i]])
        s_prime = x_dash/norm(x_dash)
        H[:, [i]] = s_prime
    H = H.T
    # Solve Total Least Square H*n = 1
    sol = np.linalg.lstsq(H, np.ones(np.size(edge_coor, 1)), rcond=None)
    n = sol[0]
    # Compute position estimation
    D_inv = np.array([[a, 0, 0],
                      [0, b, 0],
                      [0, 0, c]])
    # Compute relative position vector in Camera Frame (Moon relative to spacecraft, because a minus sign was removed)
    r_C = 1/np.sqrt(np.dot(n.T, n) - 1)*np.dot(np.dot(R_c2PA.T, D_inv), n)
    # Transform in ME frame and include the mass/geometric centers offset (move from geometric center (observed) to mass center, by moving along negative offset)
    r_C_me = np.dot(R_c2ME, r_C) - np.array([-1.736, -0.723, 0.226])
    # Return to Camera Frame
    r_C_new = np.dot(R_ME2c, r_C_me)
    range_C = norm(r_C_new)

    # print("Christian Robinson --- %s seconds ---" % (time.time() - start_time))

    return r_C_new, range_C

# ---------------------### Canny Detection + Delete False Edges ### -----------------------------


def detect_edge(img, R_ib, sun_vec_i):
    ### -------- INPUTS -------- ###
    # img:              Image Array as cv2.imread('name.png')
    # R_ib:             Rotational Matrix from Inertial to Body Frame
    # sun_vec_i:        Sun Vector (from Sun to Moon) in Inertial Frame
    ### -------- OUTPUTS ----------- ###
    # canny_edge:           Array with edge points (before rotation and pseudo-edge elimination)
    # new_edge_coor:        Matrix of edge in pixel coordinates (column vectors are coordinates in homogenous form [x y 1]^T)
    ### ------------------------ ###

    # Compare Direct Canny Edge
    # edges = canny_edge(img, 3)
    # cv2.imwrite('edges_canny.png', edges)

    start_time = time.time()
    # Rotate the Raw Image
    sun_vec_b = np.dot(R_ib, sun_vec_i)
    sun_vec_plane = np.hstack([sun_vec_b[0], -sun_vec_b[2]])
    rot_angle = np.arctan2(sun_vec_plane[1], sun_vec_plane[0])
    # print(rot_angle)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = ndimage.rotate(img_gray, rot_angle/pi*180, reshape=False)
    # cv2.imwrite('rot_gray.png', img_gray)

    # Make image into Binary (0 or 1)
    ret, img_rot = cv2.threshold(img_gray, 75, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('rot_bin.png', img_rot)

    # Apply Canny
    img_blur = cv2.medianBlur(img_rot, 3)
    canny_edge = cv2.Canny(img_blur, 127, 255)
    # cv2.imwrite('canny.png', canny_edge)

    # Extract First Edge Coordinate of each Row
    rot_edge_coor = np.nonzero(canny_edge)
    rot_edge_coor = np.transpose(rot_edge_coor)
    # Drop coordinates with duped row values (keep first row edge)
    df = pd.DataFrame(rot_edge_coor, columns=['row', 'col'])
    df.drop_duplicates(subset=['row'], inplace=True)
    window = round(df.shape[0]*0.02)
    if window < 10:
        window = 10
    df['median'] = df['col'].rolling(window).median()
    df['std'] = df['col'].rolling(window).std()
    df = df[(df.col <= df['median']+3*df['std']) &
            (df.col >= df['median']-3*df['std'])]
    df.drop(df.columns[-2:], axis=1, inplace=True)
    # error
    # Return from Pandas to Numpy
    new_rot_edge = pd.DataFrame.to_numpy(df)
    # Exclued first and last 10% (points close to terminator)
    slice_percent = math.ceil(0.03*len(new_rot_edge))
    new_rot_edge = new_rot_edge[slice_percent:-slice_percent]
    # Make from (row, col) as output of np.nonzero to (col, row) = (x, y) in pixel plane
    new_rot_edge[:, [0, 1]] = new_rot_edge[:, [1, 0]]

    # Transform from Pixel Index (int) to Pixel Coordinate (+ 0.5) and move the Center of Image
    new_rot_edge_coor = new_rot_edge + 0.5
    new_rot_edge_coor[:, [0]] -= 1024
    new_rot_edge_coor[:, [1]] -= 768
    # Rotate Edge Coordinates in Original Image Orientation
    rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                        [np.sin(rot_angle), np.cos(rot_angle)]])
    new_edge_coor = np.dot(rot_mat, new_rot_edge_coor.T).T
    # Apply opposite offset to return in Pixel Frame (0, 0) in top left corner
    new_edge_coor[:, [0]] += 1024
    new_edge_coor[:, [1]] += 768

    # Plot Edge Coordinates on Original Raw Image
    # plt.figure(1)
    # plt.tight_layout()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(img)
    # plt.plot(new_edge_coor[:, [0]]-0.5, new_edge_coor[:, [1]]-0.5,'-r', linewidth=2)
    # # plt.plot(new_rot_edge[:, [0]]-0.5, new_rot_edge[:, [1]]-0.5,'g', linewidth=1)
    # plt.savefig('img_edge.pdf')
    # # plt.show()

    # Form Matrix of Edge Coordinates
    onevec = np.ones((len(new_edge_coor), 1))
    new_edge_coor = np.concatenate((new_edge_coor, onevec), axis=1)
    new_edge_coor = np.transpose(new_edge_coor)
    # print("Edge Detection --- %s seconds ---" % (time.time() - start_time))

    return canny_edge, new_edge_coor


def wraplatitude(latdeg, londeg):
    flat = np.abs(latdeg)
    pideg = 180.

    if np.any(flat > pideg):
        latdeg[flat > pideg] = np.mod(
            latdeg[flat > pideg]+pideg, 2*pideg)-pideg
        # Set lat_wrapped flag to true if necessary
        flat = np.abs(latdeg)

    # Determine if any latitudes need to be wrapped
    idx = flat > pideg/2

    if np.any(idx):
        # Adjustments for -pi to pi
        flat = np.abs(latdeg)
        latp2 = flat > pideg/2
        londeg[idx] = londeg[idx] + pideg
        latdeg[latp2] = np.sign(latdeg(latp2))*(pideg/2-(flat[latp2]-pideg/2))
    return latdeg, londeg


def hypot(a, b):
    return np.sqrt(a**2+b**2)


def cylindrical2geodetic(rho, z, a, f):
    # Spheroid properties
    b = (1-f)*a       # Semiminor axis
    e2 = f*(2-f)      # Square of (first) eccentricity
    ae2 = a*e2
    bep2 = b*e2/(1-e2)   # b * (square of second eccentricity)

    # Starting value for parametric latitude (beta), following Bowring 1985
    r = hypot(rho, z)
    u = a*rho                    # vs. u = b * rho (Bowring 1976)
    v = b*z*(1 + bep2/r)   # vs. v = a * z   (Bowring 1976)
    cosbeta = np.sign(u)/hypot(1, v/u)
    sinbeta = np.sign(v)/hypot(1, u/v)

    # Fixed-point iteration with Bowring's formula
    # (typically converges within three iterations or less)
    count = 0
    iterate = True
    while iterate and count < 5:
        cosprev = cosbeta
        sinprev = sinbeta
        u = rho - ae2 * cosbeta**3
        v = z + bep2 * sinbeta**3
        au = a*u
        bv = b*v
        cosbeta = np.sign(au)/hypot(1, bv/au)
        sinbeta = np.sign(bv)/hypot(1, au/bv)
        iterate = np.any(hypot(cosbeta - cosprev, sinbeta - sinprev)
                         > np.finfo(float).eps)
        count = count + 1

    # Final latitude in degrees
    gdrad = np.arctan2(v, u)
    cosphi = np.cos(gdrad)
    sinphi = np.sin(gdrad)
    gd = gdrad*180/pi

    # Ellipsoidal height from final value for geodetic latitude
    N = a/np.sqrt(1 - e2*sinphi**2)
    h = rho*cosphi + (z + e2*N*sinphi)*sinphi - N

    return gd, h


def geoc2geod(gc, r, f, R):
    # Same as MATLAB geoc2geod function
    # gc in degrees (geocentric latitude)
    gc, dummy = wraplatitude(gc, gc*0)

    rho = np.cos(gc*pi/180)*r
    z = np.sin(gc*pi/180)*r

    gd, h = cylindrical2geodetic(rho.tolist(), z.tolist(), R, f)
    return gd, h


def geodetic2cylindrical(phideg, h, a, f):
    phi = phideg*pi/180
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    e2 = f*(2 - f)
    N = a/np.sqrt(1 - e2 * sinphi**2)
    rho = (N + h)*cosphi
    z = (N*(1 - e2) + h)*sinphi

    return rho, z


def geod2geoc(gd, h, f, R):
    # Same as MATLAB geod2geoc function
    # gd in degrees (geodetic latitude)
    gd, dummy = wraplatitude(gd, gd*0)

    rho, z = geodetic2cylindrical(gd, h, R, f)

    # Determine geocentric latitude and radii
    gcrad = np.arctan2(z, rho)
    gc = gcrad*180/pi
    r = np.sqrt(rho**2 + z**2)

    return gc, r


def ellipse_on_ellipsoid(lat_cdeg, lon_cdeg, ae, be, angledeg):
    # Inputs:
    # latitude and longitude of crater center, semi-axes of crater, and angle wrt horizontal [km, deg]
    # References:
    # [1] A New Global Database of Lunar Impact Craters
    # [2] Direct and Inverse Solutions of Geodesics on the Ellipsoid with Applications of Nested Equations
    # [3] https://www.movable-type.co.uk/scripts/latlong-vincenty.html
    # [4] https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    # semi-axes of Moon, km (from [1], assumed oblate sphere)
    a = 1737.1513 #should be 1737.8981
    b = 1737.1513 #assume perfect sphere [1], should be 1735.6576

    # flattening
    f = (a-b)/a  # Oblateness
    # e2 = 1-(b/a)**2 #Eccentricity

    # latitude and longitude of crater, rad
    lat_c = lat_cdeg*pi/180
    lon_c = lon_cdeg*pi/180

    # # semi-axes of crater ellipse, km
    # ae = 500
    # be = 300

    # angle of crater wrt horizontal, rad
    # angle = 30*pi/180
    angle = angledeg*pi/180

    # angle describing ellipse, rad
    theta = np.vstack(np.linspace(0, 2*pi, 40))  # column vector

    # points along ellipse, on ellipse axes
    xe = ae*np.cos(theta)
    ye = be*np.sin(theta)

    # points along ellipse, on local axes
    xp = xe*np.cos(angle) - ye*np.sin(angle)
    yp = xe*np.sin(angle) + ye*np.cos(angle)

    # figure
    # plot(xp,yp,'k-')
    # axis square
    # axis equal

    # Direct solution from [2]

    # initial azimuth, defined clockwise from north
    # this is a vector, one element per point around ellipse
    alpha1 = np.remainder(np.arctan2(xp, yp)+2*pi, 2*pi)
    cos_alpha1 = np.cos(alpha1)  # this is a vector
    sin_alpha1 = np.sin(alpha1)  # this is a vector

    # distance
    s = np.sqrt(xp**2+yp**2)  # this is a vector

    # N = a/np.sqrt(1-e2*np.sin(lat_c)**2) #row vector, one element per crater
    # coordinates of center of crater, km
    # coordinates_center = np.hstack(
    #     (a*np.cos(lat_c)*np.cos(lon_c), a*np.cos(lat_c)*np.sin(lon_c), b*np.sin(lat_c)))
    # coordinates_center = np.hstack(
    #     (N*np.cos(lat_c)*np.cos(lon_c), N*np.cos(lat_c)*np.sin(lon_c), (1-e2)*N*np.sin(lat_c))) #[3xnum_craters]
    coordinates_center = np.hstack(
        (a*np.cos(lat_c)*np.cos(lon_c), a*np.cos(lat_c)*np.sin(lon_c), a*np.sin(lat_c))) #[3xnum_craters], assume perfect sphere [1]
    # geodlat_cdeg, dummy = geoc2geod(
    #     lat_c*180/pi, np.linalg.norm(coordinates_center), f, a)
    # geodlat_c = geodlat_cdeg*pi/180
    geodlat_c = lat_c

    tan_U1 = (1-f)*np.tan(geodlat_c)  # associated to center of crater
    cos_U1 = 1/np.sqrt(1+tan_U1**2)
    sin_U1 = tan_U1*cos_U1
    sigma1 = np.arctan2(tan_U1, cos_alpha1)  # this is a vector
    sin_alpha = cos_U1*sin_alpha1  # this is a vector
    sin_alpha_sq = sin_alpha**2  # this is a vector
    cos_alpha_sq = 1-sin_alpha_sq  # this is a vector
    u2 = cos_alpha_sq*((a**2-b**2)/b**2)  # this is a vector
    A = 1 + u2/16384*(4096 + u2*(-768 + u2*(320 - 175*u2)))  # this is a vector
    B = u2/1024*(256 + u2*(-128 + u2*(74 - 47*u2)))  # this is a vector

    sigma = s/(b*A)  # initial guess, this is a vector
    for i in range(len(s)):  # for all points along ellipse
        sigma1i = sigma1[i]
        Bi = B[i]
        sigma0 = sigma[i]  # initial guess
        sigmai = sigma0  # current value
        diff_sigma = np.inf  # update
        while diff_sigma > 1e-12:
            error_i = diff_sigma
            cos_2sigmam = np.cos(2*sigma1i+sigmai)
            cos_2sigmam_sq = cos_2sigmam**2
            sin_sigma = np.sin(sigmai)
            delta_sigma = Bi*sin_sigma*(cos_2sigmam + Bi/4*(np.cos(sigmai)*(-1 + 2*cos_2sigmam_sq) -
                                        Bi/6*cos_2sigmam*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigmam_sq)))  # update
            sigmai_new = sigma0+delta_sigma  # update current value
            diff_sigma = abs(sigmai_new-sigmai)
            sigmai = sigmai_new
        sigma[i] = sigmai  # store solution for sigma
    cos_sigma = np.cos(sigma)  # this is a vector
    sin_sigma = np.sin(sigma)  # this is a vector
    cos_2sigmam = np.cos(2*sigma1+sigma)  # this is a vector
    cos_2sigmam_sq = cos_2sigmam**2  # this is a vector
    lam = np.arctan2(sin_sigma*sin_alpha1, (cos_U1*cos_sigma -
                     sin_U1*sin_sigma*cos_alpha1))  # this is a vector
    C = f/16*cos_alpha_sq*(4 + f*(4 - 3*cos_alpha_sq))  # this is a vector

    geodlat_points = np.arctan2((sin_U1*cos_sigma + cos_U1*sin_sigma*cos_alpha1), ((1-f)*np.sqrt(
        sin_alpha_sq + (sin_U1*sin_sigma - cos_U1*cos_sigma*cos_alpha1)**2)))  # this is a vector

    # Geographic coordinates
    lon_points = lon_c + (lam - (1-C)*f*sin_alpha*(sigma + C*sin_sigma *
                          (cos_2sigmam + C*cos_sigma*(-1 + 2*cos_2sigmam_sq))))  # this is a vector
    
    lon_points[lon_points > pi] = lon_points[lon_points > pi] - 2*pi
    lon_points[lon_points < -pi] += 2*pi
    
    # lat_pointsdeg, dummy = geod2geoc(geodlat_points*180/pi, 0., f, a)
    # lat_points = lat_pointsdeg*pi/180
    lat_points = geodlat_points

    # N_points = a/np.sqrt(1-e2*np.sin(lat_points)**2) #row vector, one element per crater

    # coordinates of crater rim, km
    # coordinates_rim = np.hstack((a*np.cos(lat_points)*np.cos(lon_points),
    #                              a*np.cos(lat_points)*np.sin(lon_points), b*np.sin(lat_points)))
    # coordinates_rim = np.hstack((N_points*np.cos(lat_points)*np.cos(lon_points),
    #                             N_points*np.cos(lat_points)*np.sin(lon_points), (1-e2)*N_points*np.sin(lat_points)))    
    coordinates_rim = np.hstack((a*np.cos(lat_points)*np.cos(lon_points),
                                 a*np.cos(lat_points)*np.sin(lon_points), a*np.sin(lat_points))) #assume perfect sphere [1]

    # plot ellipsoid and ellipse on surface
    # figure
    # ellipsoid(0,0,0,a,a,b)
    # hold on
    # plot3(coordinates_rim(:,1),coordinates_rim(:,2),coordinates_rim(:,3),'r')
    # hold on
    # plot3(coordinates_center(:,1),coordinates_center(:,2),coordinates_center(:,3),'ro')
    # axis equal

    # Plot latitude of points along ellipse
    # plt.figure(figsize=(9.5, 5.5))
    # plt.plot(lon_points*180/pi, lat_points*180/pi)
    return coordinates_center, coordinates_rim, lat_points, lon_points

def ellipse_on_ellipsoid_vec(lat_cdeg, lon_cdeg, ae, be, angledeg):
    # Inputs:
    # latitude and longitude of crater center, semi-axes of crater, and angle wrt horizontal [km, deg]
    # References:
    # [1] A New Global Database of Lunar Impact Craters
    # [2] Direct and Inverse Solutions of Geodesics on the Ellipsoid with Applications of Nested Equations
    # [3] https://www.movable-type.co.uk/scripts/latlong-vincenty.html
    # [4] https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    # semi-axes of Moon, km (from [1], assumed oblate sphere)
    a = 1737.1513 #should be 1737.8981
    b = 1737.1513 #from [1], it is assumed the Moon is a perfect sphere. Polar radius should be 1735.6576

    # flattening
    f = (a-b)/a  # Oblateness
    # e2 = 1-(b/a)**2 #Eccentricity

    #Make sure inputs are row vectors
    lat_cdeg = np.hstack((lat_cdeg.flatten()))
    lon_cdeg = np.hstack((lon_cdeg.flatten()))    
    ae = np.hstack((ae.flatten()))
    be = np.hstack((be.flatten()))    
    angledeg = np.hstack((angledeg.flatten()))

    # latitude and longitude of crater, rad
    lat_c = lat_cdeg*pi/180
    lon_c = lon_cdeg*pi/180

    num_craters = len(lat_cdeg)

    # angle of crater wrt horizontal, rad
    angle = angledeg*pi/180

    # angle describing ellipse, rad
    num_points = 40
    theta = np.vstack(np.linspace(0, 2*pi, num_points))  # column vector

    # points along ellipse, on ellipse axes
    xe = np.cos(theta)*ae #column vector times row vector [num_pointsx1 x 1xnum_craters] = [num_points x num_craters]
    ye = np.sin(theta)*be

    # points along ellipse, on local axes
    xp = xe*np.cos(angle) - ye*np.sin(angle) #[num_points x num_craters]
    yp = xe*np.sin(angle) + ye*np.cos(angle)

    # figure
    # plot(xp,yp,'k-')
    # axis square
    # axis equal

    # Direct solution from [2]

    # initial azimuth, defined clockwise from north
    # this is a matrix, one element per point around ellipse (column), and each row per crater
    alpha1 = np.remainder(np.arctan2(xp, yp)+2*pi, 2*pi)
    cos_alpha1 = np.cos(alpha1)  # this is a matrix
    sin_alpha1 = np.sin(alpha1)  # this is a matrix

    # distance
    s = np.sqrt(xp**2+yp**2)  # this is a matrix

    # N = a/np.sqrt(1-e2*np.sin(lat_c)**2) #row vector, one element per crater

    # coordinates of center of crater, km, https://journals.pan.pl/Content/98324/PDF/art05.pdf
    # coordinates_center = np.vstack(
    #     (N*np.cos(lat_c)*np.cos(lon_c), N*np.cos(lat_c)*np.sin(lon_c), (1-e2)*N*np.sin(lat_c))) #[3xnum_craters]
    coordinates_center = np.vstack(
        (a*np.cos(lat_c)*np.cos(lon_c), a*np.cos(lat_c)*np.sin(lon_c), a*np.sin(lat_c))) #[3xnum_craters], assume perfect sphere [1]

    # geodlat_cdeg, dummy = geoc2geod(
    #     lat_c*180/pi, np.linalg.norm(coordinates_center), f, a)
    # geodlat_c = geodlat_cdeg*pi/180
    geodlat_c = lat_c

    tan_U1 = (1-f)*np.tan(geodlat_c)  # associated to center of crater
    cos_U1 = 1/np.sqrt(1+tan_U1**2)
    sin_U1 = tan_U1*cos_U1
    sigma1 = np.arctan2(tan_U1, cos_alpha1)  # this is a matrix
    sin_alpha = cos_U1*sin_alpha1  # row vector times matrix, this is a matrix
    sin_alpha_sq = sin_alpha**2  # this is a matrix
    cos_alpha_sq = 1-sin_alpha_sq  # this is a matrix
    u2 = cos_alpha_sq*((a**2-b**2)/b**2)  # this is a matrix
    A = 1 + u2/16384*(4096 + u2*(-768 + u2*(320 - 175*u2)))  # this is a matrix
    B = u2/1024*(256 + u2*(-128 + u2*(74 - 47*u2)))  # this is a matrix

    sigma0 = s/(b*A)  # initial guess, this is a matrix
    sigma = sigma0
    diff_sigma = np.inf*np.ones((np.size(sigma,0),np.size(sigma,1))).flatten()  # update
    #Select elements that have not converged
    not_converged = np.where(diff_sigma>1e-12)[0]
    sigma1i = sigma1.flatten()
    sigmai = sigma.flatten()
    sigma0_short = sigma0.flatten()
    Bi = B.flatten()
    while len(not_converged)>0:
        cos_2sigmam = np.cos(2*sigma1i+sigmai)
        cos_2sigmam_sq = cos_2sigmam**2
        sin_sigma = np.sin(sigmai)
        delta_sigma = Bi*sin_sigma*(cos_2sigmam + Bi/4*(np.cos(sigmai)*(-1 + 2*cos_2sigmam_sq) -
                                    Bi/6*cos_2sigmam*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigmam_sq)))  # update
        sigmai_new = sigma0_short+delta_sigma  # update current value
        diff_sigma = abs(sigmai_new-sigmai)
        sigmai = sigmai_new

        check_notconverged = diff_sigma>1e-12 #check which elements have converged
        if np.any(~check_notconverged): #if any has converged
            idx_converged = not_converged[~check_notconverged]
            sigma[np.unravel_index(idx_converged, sigma.shape)] = sigmai[~check_notconverged] #assign elements that have converged
            #Continue only with elements that have not converged
            not_converged = not_converged[check_notconverged]            
            sigma1i = sigma1i[check_notconverged]
            sigma0_short = sigma0_short[check_notconverged]  # initial guess
            Bi = Bi[check_notconverged]
            sigmai = sigmai[check_notconverged]  # current value
    cos_sigma = np.cos(sigma)  # this is a matrix
    sin_sigma = np.sin(sigma)  # this is a matrix
    cos_2sigmam = np.cos(2*sigma1+sigma)  # this is a matrix
    cos_2sigmam_sq = cos_2sigmam**2  # this is a matrix
    lam = np.arctan2(sin_sigma*sin_alpha1, (cos_U1*cos_sigma -
                     sin_U1*sin_sigma*cos_alpha1))  # this is a matrix
    C = f/16*cos_alpha_sq*(4 + f*(4 - 3*cos_alpha_sq))  # this is a matrix

    geodlat_points = np.arctan2((sin_U1*cos_sigma + cos_U1*sin_sigma*cos_alpha1), ((1-f)*np.sqrt(
        sin_alpha_sq + (sin_U1*sin_sigma - cos_U1*cos_sigma*cos_alpha1)**2)))  # this is a matrix

    # Geographic coordinates
    lon_points = lon_c + (lam - (1-C)*f*sin_alpha*(sigma + C*sin_sigma *
                          (cos_2sigmam + C*cos_sigma*(-1 + 2*cos_2sigmam_sq))))  # this is a matrix

    lon_points[lon_points > pi] = lon_points[lon_points > pi] - 2*pi
    lon_points[lon_points < -pi] += 2*pi
    
    N_points = a/np.sqrt(1-e2*np.sin(geodlat_points)**2)

    # coordinates of crater rim, km
    # x_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.cos(lon_points)) #transpose, so each row represents one crater
    # y_coord_rim = np.transpose(N_points*np.cos(geodlat_points)*np.sin(lon_points))
    # z_coord_rim = np.transpose((1-e2)*N_points*np.sin(geodlat_points))
    x_coord_rim = np.transpose(a*np.cos(geodlat_points)*np.cos(lon_points)) #transpose, so each row represents one crater
    y_coord_rim = np.transpose(a*np.cos(geodlat_points)*np.sin(lon_points))
    z_coord_rim = np.transpose(a*np.sin(geodlat_points)) #assume perfect sphere [1]
    coordinates_rim = np.vstack((np.hstack((x_coord_rim.flatten())),
                                 np.hstack((y_coord_rim.flatten())), np.hstack((z_coord_rim.flatten())))) #[3x(num_points*num_craters)]

    # plot ellipsoid and ellipse on surface
    # figure
    # ellipsoid(0,0,0,a,a,b)
    # hold on
    # plot3(coordinates_rim(:,1),coordinates_rim(:,2),coordinates_rim(:,3),'r')
    # hold on
    # plot3(coordinates_center(:,1),coordinates_center(:,2),coordinates_center(:,3),'ro')
    # axis equal

    # Plot latitude of points along ellipse
    # plt.figure(figsize=(9.5, 5.5))
    # plt.plot(lon_points*180/pi, lat_points*180/pi)
    return coordinates_center, coordinates_rim, geodlat_points, lon_points, num_points, num_craters


def simulation_date(time):
    date = time  # You can also use Julian Days: date = "JD2458327.500000"
    nowET = spice.str2et(date+" TDB")  # seconds past J2000
    # J2000: determines reference frame of earth (Jan 2000)
    return nowET
# --------------------- # Input S/C position # --------------------- #
# There are 2 types of cases: looking at EARTH(1<case<100)
    # looking at MOON(case>100)


def case_type(num, training_idx=0, small_idx=0, large_idx=0, total_idx=0, state=0, days_past=0, pos_rel_earth=0, sim_time=[],month=[],day=[],hour=[]):
    case = num
    print('CASE {}'.format(case))
    # Earth cases
    if case == 1:
        # Apollo 17's Blue Marble, 12/7/1972 10AM,
        # sc distance = 29,000km
        sim_date = simulation_date("1972-12-07 10:00")
        # camera adjustments
        look_at_object = 'EARTH'
        focal_len = 25
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 268*(math.pi/180)  # 100*(math.pi/180) #theta 0<x<360
        phi = -28*(math.pi/180)  # -90<x<90
        dist = 29000 + 6371
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        sc_pos = [sc_xpos, sc_ypos, sc_zpos]
        # distance is wrong, but image is close
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp, moon_pos, sc_pos]
    elif case == 3000:
        # Apollo 15 7/26/1971
        # sp distance: 25,000-30,000 NAUTICAL MILES
        # https://moon.nasa.gov/resources/224/view-of-earth-photographed-by-apollo-15-on-voyage-to-the-moon/?category=images
        if month>9:
            year = 2023
        else:
            year = 2024
        if month<10:
            str_month = "0" + str(int(month))
        else:
            str_month = str(int(month))
        if day<10:
            str_day = "0" + str(int(day))
        else:
            str_day = str(int(day))
        if hour<10:
            str_hour = "0" + str(int(hour))
        else:
            str_hour = str(int(hour))
        sim_date = simulation_date(str(year) + "-" + str_month + "-" + str_day + " " + str_hour + ":00")
        # camera adjustments
        look_at_object = 'EARTH'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0] #longitude, rad
        phi = state[1] #latitude, rad
        dist = state[2] #km
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        sc_pos = [sc_xpos, sc_ypos, sc_zpos]
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]    
   
    elif case == 2:
        # 1990 Galileo family portrait 12/11/1990 07:50
        sim_date = simulation_date("1990-12-11 07:50")
        # camera adjustments
        look_at_object = 'EARTH'
        focal_len = 1500
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 215*(math.pi/180)  # 100*(math.pi/180) #theta 0<x<360
        phi = -34*(math.pi/180)  # -90<x<90
        dist = 2100000 + 6731
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        sc_pos = [sc_xpos, sc_ypos, sc_zpos]
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp, moon_pos, sc_pos]
    elif case == 3:
        # Apollo 15 7/26/1971
        # sp distance: 25,000-30,000 NAUTICAL MILES
        # https://moon.nasa.gov/resources/224/view-of-earth-photographed-by-apollo-15-on-voyage-to-the-moon/?category=images
        sim_date = simulation_date("2024-10-20 17:17")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'EARTH'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = 205*(math.pi/180)
        # theta = 255*(math.pi/180)
        phi = -5*(math.pi/180)
        dist = 150000+6371 #150000+6371
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        sc_pos = [sc_xpos, sc_ypos, sc_zpos]
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]
    # Moon cases
    elif case == 1000:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-21 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 20
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = 350*(math.pi/180)-210*training_idx/100*np.pi/180
        phi = -30*(math.pi/180)
        dist = 12000  # 10000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1001:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-21 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 20
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = 350*(math.pi/180)
        phi = -180*(math.pi/180)+360*(training_idx-small_idx) / \
            (large_idx-small_idx)*(math.pi/180)
        dist = 12000  # 10000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1002:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-21 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1003:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-24 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1004:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-22 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1005:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-23 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]        
    elif case == 10005:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-12-07 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]                         
    elif case == 2000:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2024-08-01 10:00")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]                        
    elif case == 2001: #very low resolution
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2024-08-01 10:00")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 1024
        cam_height = 768
        px = 4.96E-3*2
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]                     
    elif case == 2002: #very high resolution
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2024-08-01 10:00")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 4096
        cam_height = 3072
        px = 4.96E-3/2
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]                  
    elif case == 2003: #high resolution
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2024-08-01 10:00")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 3072
        cam_height = 2304
        px = 4.96E-3/1.5
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]                 
    elif case == 2004: #low resolution
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2024-08-01 10:00")
        sim_date = sim_date+days_past*3600*24
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 1536
        cam_height = 1152
        px = 4.96E-3*1.5
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        theta = state[0]
        phi = state[1]
        dist = state[2]
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]
    elif case == 101:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-11-21 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 30
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 295*(math.pi/180)
        phi = -25*(math.pi/180)
        dist = 22500  # 10000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 102:
        # NASA's Cassini 08/17/1999
        sim_date = simulation_date("1999-08-17 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 1750
        cam_width = 1084
        cam_height = 1025
        px = 0.0120
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 40*(math.pi/180)
        phi = 20*(math.pi/180)
        dist = 500000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 103:
        # orion 11/22/2022
        sim_date = simulation_date("2022-11-22 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 20
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 350*(math.pi/180)
        phi = -10*(math.pi/180)
        dist = 10000  # 527.865 + 1737.4
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 104:
        # ArgoMoon
        sim_date = simulation_date("2022-11-17 10:55")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 120
        cam_width = 4096
        cam_height = 3072
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = 310*(math.pi/180)
        phi = -25*(math.pi/180)
        dist = 278500  # 10000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 109:
        # orion flyby 11/21/22 10:00
        sim_date = simulation_date("2022-01-15 10:00")
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 30
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = 0
        reflection = 1
        # S/C Position
        theta = -80*(math.pi/180)
        phi = -10*(math.pi/180)
        dist = 22500  # 10000
        sc_xpos = dist*math.cos(theta)*math.cos(phi)
        sc_ypos = dist*math.sin(theta)*math.cos(phi)
        sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos, moon_sp[0], moon_pos, sc_pos]
    elif case == 1234:
        # orion flyby 11/21/22 10:00
        sim_date = sim_time
        # camera adjustments
        look_at_object = 'MOON'
        focal_len = 35
        cam_width = 2048
        cam_height = 1536
        px = 4.96E-3
        camera_definition = [look_at_object,
                             focal_len, cam_width, cam_height, px]
        iter = training_idx
        reflection = 1
        # S/C Position
        # sc_xpos = dist*math.cos(theta)*math.cos(phi)
        # sc_ypos = dist*math.sin(theta)*math.cos(phi)
        # sc_zpos = dist*math.sin(phi)
        moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
        moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
        sc_pos = pos_rel_earth
        sc_xpos = sc_pos[0]-moon_pos[0]
        sc_ypos = sc_pos[1]-moon_pos[1]
        sc_zpos = sc_pos[2]-moon_pos[2]
        theta = np.arctan2(sc_ypos,sc_xpos)        
        dist = np.sqrt(sc_xpos**2.+sc_ypos**2.+sc_zpos**2.)
        phi = np.arcsin(sc_zpos/dist)
        mission_specifics = [theta, phi, dist, sc_xpos,
                             sc_ypos, sc_zpos]
        
    return sim_date, camera_definition, sc_pos, iter, reflection, mission_specifics


def crater_mapping(date, mission_specifics, camera_definition, xyz_moon, center_moon, img_width, img_height, fig3d_idx, plots_yesno, test):  # points or ellipses
    # date, mission_specifics and camera_definition are from main.py and case_type function
    # mission_specifics = [theta,phi,dist,sc_xpos,sc_ypos,sc_zpos,moon_sp,moon_pos,sc_pos]
    # camera_definition = [look_at_object,focal_len,cam_width,cam_height,px]
    # type: 1 = ellipse, 2 = point
    sim_date = date
    # theta = mission_specifics[0]
    # phi = mission_specifics[1]
    # dist = mission_specifics[2]
    # sc_xpos = mission_specifics[3]
    # sc_ypos = mission_specifics[4]
    # sc_zpos = mission_specifics[5]
    # moon_sp = mission_specifics[6]
    moon_pos = mission_specifics[7]
    sc_pos = mission_specifics[8]
    # print('moon_pos = ' + str(moon_pos))
    # print('sc_pos = ',str(sc_pos))
    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]
    # ROTATIONS--------------------------------------------------------------------------
    R_bi = rot_body2inertial(sc_pos, moon_pos)
    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", sim_date)

    craters_Moon_J200 = np.dot(np.transpose(
        rotm_moon), np.transpose(np.vstack((xyz_moon, center_moon))))  # concatenate center of the crater

    craters_earth_x = craters_Moon_J200[0, :] + moon_pos[0]
    craters_earth_y = craters_Moon_J200[1, :] + moon_pos[1]
    craters_earth_z = craters_Moon_J200[2, :] + moon_pos[2]
    craters_earth = np.vstack(
        (craters_earth_x, craters_earth_y, craters_earth_z))

    craters_sc_x = craters_earth_x - sc_pos[0]
    craters_sc_y = craters_earth_y - sc_pos[1]
    craters_sc_z = craters_earth_z - sc_pos[2]
    craters_sc = np.vstack((craters_sc_x, craters_sc_y, craters_sc_z))

    craters_sun_x = craters_earth_x - sun_pos_e[0]
    craters_sun_y = craters_earth_y - sun_pos_e[1]
    craters_sun_z = craters_earth_z - sun_pos_e[2]
    craters_sun = np.vstack((craters_sun_x, craters_sun_y, craters_sun_z))

    craters_cam = np.dot(np.transpose(R_bi), craters_sc)
    X_c = craters_cam[0, :]
    Y_c = craters_cam[1, :]
    Z_c = craters_cam[2, :]

    # determining craters that face the camera
    craters_facing_cam = np.sum(craters_sc*craters_Moon_J200, axis=0)
    # determining which craters are light/dark
    craters_lit = np.sum(craters_sun*craters_Moon_J200, axis=0)

    # angle between crater center and sun vector
    angle_sun_deg = np.arccos(np.min([1, np.max([-1, -craters_lit[-1]/np.linalg.norm(
        craters_sun[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1])])]))*180/np.pi

    # spacecraft incidence angle
    #     sc_inc_angle_deg = np.arccos(np.min([1, np.max([-1, -craters_facing_cam[-1]/np.linalg.norm(craters_sc[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1])])]))*180/np.pi
    sc_inc_angle_deg = np.arccos(np.clip(-craters_facing_cam[-1]/np.linalg.norm(craters_sc[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1]), -1, 1)) * 180 / np.pi
    
    # Conversion to Pixels---------------------------------------
    # mm, #note, focal_len depends on case. Need to make this into function
    focal_len = camera_definition[1]
    px = camera_definition[4]
    cam_width = camera_definition[2]
    cam_height = camera_definition[3]
    # another for loop
    pix_x = np.array(X_c)/np.array(Y_c)*focal_len/px
    pix_z = np.array(Z_c)/np.array(Y_c)*focal_len/px
    # plot
    plot_crater_x = pix_x+cam_width/2
    plot_crater_z = cam_height/2-pix_z
    # choose only values inside camera
    if test==0: #for training, points need to be within field of view, facing the camera, and illuminated
        filter_visible = np.where((plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
            plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0))
    elif test==1: #if testing, points do not need to be illuminated, only within the field of view and facing the camera
        filter_visible = np.where((plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
            plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0))
    select_x = plot_crater_x[filter_visible[0]]
    select_z = plot_crater_z[filter_visible[0]]

    length_visible = len(filter_visible[0])

    center_x = (img_width/cam_width)*plot_crater_x[-1]  # pixel of center
    center_z = (img_height/cam_height)*plot_crater_z[-1]
    center_visible = 0
    # if last visible point is the center of the crater (it was concatenated before)
    if length_visible > 0 and filter_visible[0][-1] == len(plot_crater_x)-1:
        center_visible = 1
        select_x = select_x[:-1]  # remove center from vector
        select_z = select_z[:-1]

    visible = 1
    # if all points along crater and center are visible
    if length_visible == len(plot_crater_x):
        visible = 2

    # if no points are visible
    if length_visible == 0:
        visible = 0
        
    # scale pixel values in case image is not the expected size
    select_x = (img_width/cam_width)*select_x
    select_z = (img_height/cam_height)*select_z

    # # plotting
    # if plots_yesno==0:
    #     plt.figure(fig3d_idx, figsize=(15, 15))
    #     plt.plot(select_x, select_z, linewidth=0.5, color='red')  # for 2D
    #     if center_visible == 1 and visible == 2:  # center is visible and whole rim as well
    #         plt.plot(center_x, center_z, color='g', marker='o',
    #                 linestyle='None', markersize=2.0)
    #     elif center_visible == 1:  # center is visible but not the whole rim
    #         plt.plot(center_x, center_z, color='r', marker='o',
    #                 linestyle='None', markersize=2.0)
    # plt.savefig("moon_mapped.png",bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    return center_x, center_z, select_x, select_z, center_visible, visible, angle_sun_deg, sc_inc_angle_deg

def crater_mapping_vec(date, mission_specifics, camera_definition, xyz_moon, center_moon, img_width, img_height, fig3d_idx, plots_yesno, test, num_points, num_craters):  # points or ellipses
    # date, mission_specifics and camera_definition are from main.py and case_type function
    # mission_specifics = [theta,phi,dist,sc_xpos,sc_ypos,sc_zpos,moon_sp,moon_pos,sc_pos]
    # camera_definition = [look_at_object,focal_len,cam_width,cam_height,px]
    # type: 1 = ellipse, 2 = point
    sim_date = date
    theta = mission_specifics[0]
    phi = mission_specifics[1]
    dist = mission_specifics[2]
    sc_xpos = mission_specifics[3]
    sc_ypos = mission_specifics[4]
    sc_zpos = mission_specifics[5]
    moon_sp = mission_specifics[6]
    moon_pos = mission_specifics[7]
    sc_pos = mission_specifics[8]
    # print('moon_pos = ' + str(moon_pos))
    # print('sc_pos = ',str(sc_pos))
    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]
    # ROTATIONS--------------------------------------------------------------------------
    R_bi = rot_body2inertial(sc_pos, moon_pos)
    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", sim_date)
    craters_Moon_J200 = np.dot(np.transpose(
        rotm_moon), np.hstack((xyz_moon, center_moon)))  # concatenate center of the crater
    craters_earth = craters_Moon_J200 + np.vstack((moon_pos))
    
    craters_sc = craters_earth - np.vstack((sc_pos))
    craters_sun = craters_sc - np.vstack((sun_pos_e))
    craters_cam = np.dot(np.transpose(R_bi), craters_sc)
    X_c = craters_cam[0, :]
    Y_c = craters_cam[1, :]
    Z_c = craters_cam[2, :]
    # determining craters that face the camera
    craters_facing_cam = np.sum(craters_sc*craters_Moon_J200, axis=0)
    # determining which craters are light/dark
    craters_lit = np.sum(craters_sun*craters_Moon_J200, axis=0)
    num_pointscraters = num_craters*num_points
    craters_facing_cam_centers = craters_facing_cam[num_pointscraters:]
    craters_lit_centers = craters_lit[num_pointscraters:]
    craters_Moon_J200_centers = craters_Moon_J200[:,num_pointscraters:]
    craters_sc_centers = craters_sc[:,num_pointscraters:]
    craters_sun_centers = craters_sun[:,num_pointscraters:]
    # angle between crater center and sun vector
    angle_sun_deg = np.arccos(np.clip(-craters_lit_centers/np.sqrt(np.sum(
        craters_sun_centers**2,axis=0))/np.sqrt(np.sum(craters_Moon_J200_centers**2,axis=0)), -1, 1))*(180/np.pi)
    # spacecraft incidence angle
    #     sc_inc_angle_deg = np.arccos(np.min([1, np.max([-1, -craters_facing_cam[-1]/np.linalg.norm(craters_sc[:, -1])/np.linalg.norm(craters_Moon_J200[:, -1])])]))*180/np.pi
    sc_inc_angle_deg = np.arccos(np.clip(-craters_facing_cam_centers/np.sqrt(np.sum(
        craters_sc_centers**2,axis=0))/np.sqrt(np.sum(craters_Moon_J200_centers**2,axis=0)), -1, 1))*(180/np.pi)
    
    # Conversion to Pixels---------------------------------------
    # mm, #note, focal_len depends on case. Need to make this into function
    focal_len = int(camera_definition[1])
    px = float(camera_definition[4])
    cam_width = int(camera_definition[2])
    cam_height = int(camera_definition[3])
    # another for loop
    pix_x = np.array(X_c)/np.array(Y_c)*focal_len/px
    pix_z = np.array(Z_c)/np.array(Y_c)*focal_len/px
    # plot
    plot_crater_x = pix_x+cam_width/2
    plot_crater_z = cam_height/2-pix_z
    #Reshape: [num_craters x num_points]
    plot_crater_x = np.hstack((np.reshape(plot_crater_x[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_x[num_pointscraters:]))))
    plot_crater_z = np.hstack((np.reshape(plot_crater_z[:num_pointscraters],(num_craters,num_points)),np.vstack((plot_crater_z[num_pointscraters:]))))
    craters_facing_cam = np.hstack((np.reshape(craters_facing_cam[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_facing_cam_centers))))
    craters_lit = np.hstack((np.reshape(craters_lit[:num_pointscraters],(num_craters,num_points)),np.vstack((craters_lit_centers))))
    # choose only values inside camera
    if test==0: #for training, points need to be within field of view, facing the camera, and illuminated
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
            plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0) & (craters_lit <= 0)
    elif test==1: #if testing, points do not need to be illuminated, only within the field of view and facing the camera
        filter_visible = (plot_crater_x >= 0) & (plot_crater_x <= cam_width) & (
            plot_crater_z >= 0) & (plot_crater_z <= cam_height) & (craters_facing_cam <= 0)
    center_x = plot_crater_x[:,[-1]]  # pixel of center
    center_z = plot_crater_z[:,[-1]]    
    center_visible = np.zeros((num_craters,1))
    center_visible[filter_visible[:,-1]] = 1
    visible = np.zeros((num_craters,1))
    visible[np.any(filter_visible,axis=1)] = 1 #partially visible
    visible[np.all(filter_visible,axis=1)] = 2 #all points visible
        
    # scale pixel values in case image is not the expected size    
    plot_rim_x = plot_crater_x[:,:num_points]
    plot_rim_z = plot_crater_z[:,:num_points]
    isrim_visible = filter_visible[:,:num_points]
    # plotting
    # if plots_yesno==0:
    #     for idx_crater in range(num_craters):
    #         visible_i = visible[idx_crater]
    #         center_visible_i = center_visible[idx_crater]
    #         if visible_i>0:
    #             rim_select = isrim_visible[idx_crater,:]
    #             select_x = plot_rim_x[idx_crater,rim_select]
    #             select_z = plot_rim_z[idx_crater,rim_select]
    #             plt.figure(fig3d_idx, figsize=(15, 15))
    #             plt.plot(select_x, select_z, linewidth=0.5, color='cyan', linestyle='--')  # for 2D
    #             if center_visible_i == 1 and visible_i == 2:  # center is visible and whole rim as well
    #                 plt.plot(center_x[idx_crater], center_z[idx_crater], color='g', marker='x',
    #                         linestyle='None', markersize=2.0)
    #             elif center_visible_i == 1:  # center is visible but not the whole rim
    #                 plt.plot(center_x[idx_crater], center_z[idx_crater], color='r', marker='x',
    #                         linestyle='None', markersize=2.0)
    # # plt.savefig("moon_mapped.png",bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    return center_x, center_z, plot_rim_x, plot_rim_z, isrim_visible, center_visible, visible, angle_sun_deg, sc_inc_angle_deg

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height


def latlong2cart(latitudedeg, longitudedeg):
    # a, b, c = 1739.088, 1737.37, 1734.969    # km
    a = 1737.8981  # km, should be 1737.8981, from "A New Global Database of Lunar Impact Craters"
    b = 1737.8981  # km, should be 1737.8981
    c = 1735.6576  # km

    lat = latitudedeg*np.pi/180
    lon = longitudedeg*np.pi/180
    cos_lat = np.cos(lat)
    x = a*cos_lat*np.cos(lon)
    y = b*cos_lat*np.sin(lon)
    z = c*np.sin(lat)
    return np.hstack((x, y, z))

# ---------------------### Sobel over 5x5 mask ### -----------------------------


def drawLine(n, m, slope, pixel_start, num_parallel, step_size):
    #
    # Draws lines parallel to a given slope (nominally the illumination vector)
    # Returns pixel indices of drawn lines
    #
    ### -------- INPUTS -------- ###
    # n:                Image height in pixels
    # m:                Image width in pixels
    # slope:            Slope of desired lines (slope of illumniation vector)
    # pixel_start:      Horizontal offset of the center line eminating from the top border
    # num_parallel:     Number of lines parallel to center line
    # step_size:        Seperation between parallel lines in pixels
    ### -------- OUTPUTS ----------- ###
    # pixel_indices:    Pixel indices of drawn lines
    # image:            Numpy array image to visualize scan lines
    ### ------------------------ ###

    image = np.zeros((n, m))
    pixel_indices = []

    if slope == 0:  # draw horizontal lines
        # Iterate for each line
        for num in range(int(pixel_start-step_size*num_parallel/2), int(pixel_start+step_size*num_parallel/2)+step_size, step_size):
            # Draw from leftmost line to rightmost
            pixel_start_p = num
            # Define column vector and row as a function of columns
            rowvec = np.linspace(0, m-1, m)
            colvec = pixel_start_p*np.ones(m)  # vertical pixel indicies
            # Filter invalid columns
            np_mask3 = colvec[:] > n-1
            colvec_nan = colvec[np_mask3]
            colvec_nan[:] = np.nan
            colvec[np_mask3] = colvec_nan
            np_mask5 = colvec[:] < 0
            colvec_nan1 = colvec[np_mask5]
            colvec_nan1[:] = np.nan
            colvec[np_mask5] = colvec_nan1
            # Filter invalid rows
            # rowvec = rowvec - rowvec[-1]
            np_mask1 = rowvec[:] > m-1
            rowvec_nan = rowvec[np_mask1]
            rowvec_nan[:] = np.nan
            rowvec[np_mask1] = rowvec_nan
            np_mask4 = rowvec[:] < 0
            rowvec_nan1 = rowvec[np_mask4]
            rowvec_nan1[:] = np.nan
            rowvec[np_mask4] = rowvec_nan1
            # Combine rows and columns for coords
            line_coords = np.transpose(np.vstack((rowvec, colvec)))
            # Draw all valid coordinates
            np_mask2 = np.isnan(line_coords).any(axis=1)
            line_coords = np.intc(line_coords[~np_mask2])
            image[line_coords[:, 1], line_coords[:, 0]] = 255
            # Append 'coords' to list of indices
            pixel_indices.append(line_coords)
        cv2.imwrite('line_mask_test.png', image)
        return pixel_indices, image

    if (slope >= 0 and slope <= 1) or (slope >= -1 and slope < 0):  # draw from left border
        # Iterate for each line
        pixel_start = np.ceil(slope*(pixel_start))-1
        for num in range(int(pixel_start-step_size*num_parallel/2), int(pixel_start+step_size*num_parallel/2)+step_size, step_size):
            # Draw from leftmost line to rightmost
            pixel_start_p = num
            # Define row vector and columns as a function of rows
            #rowvec = np.linspace(0,pixel_start_p,pixel_start_p+1)
            rowvec = np.linspace(0, m-1, m)
            #print('rowvec', rowvec)
            colvec = np.floor(slope*rowvec)  # vertical pixel indicies
            colvec = pixel_start_p-colvec
            # Filter invalid rows
            np_mask3 = rowvec[:] > m-1
            rowvec_nan = rowvec[np_mask3]
            rowvec_nan[:] = np.nan
            rowvec[np_mask3] = rowvec_nan
            # Filter invalid columns
            #colvec = colvec - colvec[-1]
            np_mask1 = colvec[:] > n-1
            colvec_nan = colvec[np_mask1]
            colvec_nan[:] = np.nan
            colvec[np_mask1] = colvec_nan
            np_mask4 = colvec[:] < 0
            colvec_nan1 = colvec[np_mask4]
            colvec_nan1[:] = np.nan
            colvec[np_mask4] = colvec_nan1
            # Combine rows and columns for coords
            line_coords = np.transpose(np.vstack((rowvec, colvec)))
            # Draw all valid coordinates
            np_mask2 = np.isnan(line_coords).any(axis=1)
            line_coords = np.intc(line_coords[~np_mask2])
            image[line_coords[:, 1], line_coords[:, 0]] = 255
            # Append 'coords' to list of indices
            pixel_indices.append(line_coords)
        cv2.imwrite('line_mask_test.png', image)
        return pixel_indices, image

    elif slope > 1 or slope < -1:  # draw from top border
        # Iterate for each line
        for num in range(int(pixel_start-step_size*num_parallel/2), int(pixel_start+step_size*num_parallel/2)+step_size, step_size):
            # Draw from leftmost line to rightmost
            pixel_start_p = num
            # Define column vector and row as a function of columns
            colvec = np.linspace(0, n-1, n)
            rowvec = np.floor((1/slope)*colvec)  # horizontal pixel indicies
            rowvec = pixel_start_p-rowvec  # missing this step might be bug in other code
            # Filter invalid columns
            np_mask3 = colvec[:] > n-1
            colvec_nan = colvec[np_mask3]
            colvec_nan[:] = np.nan
            colvec[np_mask3] = colvec_nan
            # Filter invalid rows
            # rowvec = rowvec - rowvec[-1]
            np_mask1 = rowvec[:] > m-1
            rowvec_nan = rowvec[np_mask1]
            rowvec_nan[:] = np.nan
            rowvec[np_mask1] = rowvec_nan
            np_mask4 = rowvec[:] < 0
            rowvec_nan1 = rowvec[np_mask4]
            rowvec_nan1[:] = np.nan
            rowvec[np_mask4] = rowvec_nan1
            # Combine rows and columns for coords
            line_coords = np.transpose(np.vstack((rowvec, colvec)))
            # Draw all valid coordinates
            np_mask2 = np.isnan(line_coords).any(axis=1)
            line_coords = np.intc(line_coords[~np_mask2])
            image[line_coords[:, 1], line_coords[:, 0]] = 255
            # Append 'coords' to list of indices
            pixel_indices.append(line_coords)
        cv2.imwrite('line_mask_test.png', image)
        return pixel_indices, image

    else:  # vertical
        # Iterate for each line
        for num in range(int(pixel_start-step_size*num_parallel/2), int(pixel_start+step_size*num_parallel/2)+step_size, step_size):
            # Draw from leftmost line to rightmost
            pixel_start_p = num
            # Define column vector and row as a function of columns
            colvec = np.linspace(0, n-1, n)
            rowvec = pixel_start_p*np.ones(n)  # horizontal pixel indicies
            # Filter invalid columns
            np_mask3 = colvec[:] > n-1
            colvec_nan = colvec[np_mask3]
            colvec_nan[:] = np.nan
            colvec[np_mask3] = colvec_nan
            np_mask5 = colvec[:] < 0
            colvec_nan1 = colvec[np_mask5]
            colvec_nan1[:] = np.nan
            colvec[np_mask5] = colvec_nan1
            # Filter invalid rows
            # rowvec = rowvec - rowvec[-1]
            np_mask1 = rowvec[:] > m-1
            rowvec_nan = rowvec[np_mask1]
            rowvec_nan[:] = np.nan
            rowvec[np_mask1] = rowvec_nan
            np_mask4 = rowvec[:] < 0
            rowvec_nan1 = rowvec[np_mask4]
            rowvec_nan1[:] = np.nan
            rowvec[np_mask4] = rowvec_nan1
            # Combine rows and columns for coords
            line_coords = np.transpose(np.vstack((rowvec, colvec)))
            # Draw all valid coordinates
            np_mask2 = np.isnan(line_coords).any(axis=1)
            line_coords = np.intc(line_coords[~np_mask2])
            image[line_coords[:, 1], line_coords[:, 0]] = 255
            # Append 'coords' to list of indices
            pixel_indices.append(line_coords)
        cv2.imwrite('line_mask_test.png', image)
        return pixel_indices, image


def scanImage(img, pixel_indices, slope, sun_vec):
    #
    # Scans input image given pixel indices in order of line, line pixel index
    # for intercepts
    # Returns pixel indices of edge intercepts
    #
    ### -------- INPUTS -------- ###
    # img:                  Image to scan, must be a binary numpy array
    # pixel_indices:        Pixel indices of lines to scan
    # slope:                Slope of pixel_indices lines (slope of illumniation vector)
    # sun_vec:              Illumination vector
    ### -------- OUTPUTS ----------- ###
    # intercept_indices:    Pixel indices of edge intercepts
    ### ------------------------ ###

    n = np.shape(img)[0]
    m = np.shape(img)[1]
    image = np.zeros((n, m))
    intercept_indices = []

    for line in pixel_indices:
        # check scan direction for every slope case    
        if slope == 0:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > 0 and slope <= 1:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > 1 and slope < 1000000000000:
            if sun_vec[0] > 0:
                line = line
            else:
                line = np.flipud(line)
        elif slope < -1 and slope > -1000000000000:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > -1 and slope < 0:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        else:
            if sun_vec[1] > 0:
                line = line
            else:
                line = np.flipud(line)

        new_row = 0*line[:, 1]
        new_row = np.array(new_row, dtype=bool)
        p = 0
        three_tuple = [(current,
                        line[idx + 1] if idx < len(line) - 1 else None,
                        line[idx + 2] if idx < len(line) - 2 else None) for idx, current in enumerate(line)]
        for tuple in three_tuple:  # if 3 consecutive non-black pixels, set first pixel to white
            # more efficient?
            i0 = tuple[0][0]
            j0 = tuple[0][1]
            if img[j0][i0] > 0:
                if tuple[1] is not None:
                    i1 = tuple[1][0]
                    j1 = tuple[1][1]
                    if img[j1][i1] > 0:  # next element in line, not global +1
                        if tuple[2] is not None:
                            i2 = tuple[2][0]
                            j2 = tuple[2][1]
                            if img[j2][i2] > 0:
                                new_row[p] = True
                                break  # don't scan reverse to get rid of terminator
                                q = -1
                                # scan in reverse direction
                                for i, j in np.flipud(line):
                                    if img[j][i] > 0:
                                        if img[j+1][i+1] > 0:
                                            if img[j+2][i+2] > 0:
                                                new_row[q] = True
                                                break  # move onto next line
                                            else:
                                                q += -1
                                                continue
                                        else:
                                            q += -1
                                            continue
                                    else:
                                        q += -1
                                        continue
                            else:
                                p += 1
                                continue
                        else:
                            break
                    else:
                        p += 1
                        continue
                else:
                    break
            else:
                p += 1
                continue
        intercepts = line*new_row[:, None]
        nonzero_intercepts = intercepts[(
            intercepts != np.array([0, 0])).any(axis=1)]
        image[nonzero_intercepts[:, 1], nonzero_intercepts[:, 0]] = 255
        intercept_indices.append(nonzero_intercepts)
        intercept_indices = [y for y in intercept_indices if 0 not in y.shape]
    cv2.imwrite('scanImageTest.png', image)
    return intercept_indices


def generateGrid(primary_indices):
    #
    # Creates 5x5 grids of pixel indices about given primary pixel indices
    # Returns grids' indices
    #
    ### -------- INPUTS -------- ###
    # primary_indices:      Pixel indices to generate grids about
    ### -------- OUTPUTS ----------- ###
    # grids:                Pixel indices of 5x5 grids
    ### ------------------------ ###

    grids = np.zeros((2, 2))
    # define the x and y offsets
    offset = np.arange(-2, 3)
    # create a meshgrid of the offsets
    x_mesh, y_mesh = np.meshgrid(offset, offset)
    offsets = np.vstack((x_mesh.ravel(), y_mesh.ravel())).T
    for point in primary_indices:
        grid = offsets+point
        grids = np.concatenate((grids, grid), axis=0)
    grids = grids[2:, :]
    return grids


def spaceLine(img, intercept_indices, n, m, slope, pixel_start, num_parallel, step_size, sun_vec, count=0):
    #
    # Recursively spaces lines until a specified threshold of repeated pixels is met
    # Desire 0 < repeated pixels < threshold such that 5x5 grids just barely overlap
    # Minimum step size is 2 pixels
    # Returns step size in pixels, input for drawLine
    #
    ### -------- INPUTS -------- ###
    # img:                  Image to scan, must be a binary numpy array
    # intercept_indices:    Pixel indices of edge intercepts
    # n:                    Image height in pixels
    # m:                    Image width in pixels
    # slope:                Slope of desired lines (slope of illumniation vector)
    # pixel_start:          Horizontal offset of the center line eminating from the top border
    # num_parallel:         Number of lines parallel to center line
    # step_size:            Seperation between parallel lines in pixels
    # sun_vec:              Illumination vector
    ### -------- OUTPUTS ----------- ###
    # step_size:            Seperation between parallel lines in pixels
    ### ------------------------ ###

    k = 4  # 4  #2
    threshold = k*len(intercept_indices)
    #print('threshold: ',threshold)
    num_points = len(intercept_indices)*25
    points = generateGrid(intercept_indices)
    num_unique_points = len(np.unique(points, axis=0))
    num_repeated_points = num_points-num_unique_points
    #print('numrep: ',num_repeated_points)

    if num_repeated_points <= threshold and num_repeated_points > 0:
        # print('')
        # print('converged!')
        # print('----------------')
        #print('step_size: ', step_size)
        #print('numrep: ',num_repeated_points)
        pass
    elif count > 25:
        #print('recursion count exceeded')
        pass
    else:
        if num_repeated_points == 0:
            if step_size > 2:
                step_size -= 1
                #print('step_size: ',step_size)
                indices, temp = drawLine(
                    n, m, slope, pixel_start, num_parallel, step_size)
                intercept_indices = scanImage(img, indices, slope, sun_vec)
                spaceLine(img, intercept_indices, n, m, slope,
                          pixel_start, num_parallel, step_size, sun_vec, count+1)
            else:
                step_size = 2
        else:
            step_size += 1
            #print('step_size: ',step_size)
            indices, temp = drawLine(
                n, m, slope, pixel_start, num_parallel, step_size)
            intercept_indices = scanImage(img, indices, slope, sun_vec)
            spaceLine(img, intercept_indices, n, m, slope,
                      pixel_start, num_parallel, step_size, sun_vec, count+1)
    return step_size


def offsetLine(n, m, slope):
    #
    # Calculates the offset for a given slope such that the center
    # of the line is conincident with the center of the image
    #
    ### -------- INPUTS -------- ###
    # n:                Image height in pixels
    # m:                Image width in pixels
    # slope:            Slope of given line (slope of illumniation vector)
    ### -------- OUTPUTS ----------- ###
    # pixel_start:      Horizontal offset of the center line eminating from the top border
    ### ------------------------ ###

    if slope == 0:
        pixel_start = n//2
    elif slope == np.nan:
        pixel_start = m//2
    else:
        pixel_start = m-1/2*(m-n/slope)
    return pixel_start


def numLine(img, n, m, pixel_start, slope, step_size, sun_vec):
    #
    # Finds the bounding lines of interception for a given slope
    # Returns number of parallel lines required to fill gap between
    # bounding lines given a step size
    #
    ### -------- INPUTS -------- ###
    # img:                  Image to scan, must be a binary numpy array
    # n:                    Image height in pixels
    # m:                    Image width in pixels
    # pixel_start:          Horizontal offset of the center line eminating from the top border
    # slope:                Slope of pixel_indices lines (slope of illumniation vector)
    # step_size:            Seperation between parallel lines in pixels
    # sun_vec:              Illumination vector
    ### -------- OUTPUTS ----------- ###
    # num_parallel:         Number of lines parallel to center line
    ### ------------------------ ###

    intercept_indices = [0]
    gap = step_size
    while len(intercept_indices) != 0:
        gap += 1000
        #print('gap1000: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    gap -= 1000
    intercept_indices = [0]
    while len(intercept_indices) != 0:
        gap += 300
        #print('gap300: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    gap -= 300
    intercept_indices = [0]
    while len(intercept_indices) != 0:
        gap += 100
        #print('gap100: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    gap -= 100
    intercept_indices = [0]
    while len(intercept_indices) != 0:
        gap += 30
        #print('gap25: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    gap -= 30
    intercept_indices = [0]
    while len(intercept_indices) != 0:
        gap += 10
        #print('gap10: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    gap -= 10
    intercept_indices = [0]
    while len(intercept_indices) != 0:
        gap += 3
        #print('gap3: ', gap)
        indices, temp = drawLine(n, m, slope, pixel_start, 1, gap)
        intercept_indices = scanImage(img, indices, slope, sun_vec)
    return np.ceil(gap/step_size+1)


def noiseMeanSecondary(img_gray, threshold):
    #
    # Secondary function for image noise mean and sigma calculation
    # Calculates based on all pixel values under given threshold
    #
    ### -------- INPUTS -------- ###
    # img_gray:         Single channel, grayscale image
    # threshold:        Upper threshold for pixel sample, any pixel value < threshold is sampled
    ### -------- OUTPUTS ----------- ###
    # mean:             Mean of sampled pixels
    # sigma:            Standard deviation of sampled pixels
    ### ------------------------ ###

    img_gray_noise = img_gray[img_gray < threshold]
    mean = img_gray_noise.mean()
    sigma = img_gray_noise.std()
    return mean, sigma


def noiseMean(sun_vec, img_gray, threshold=50, secondary_threshold=50):
    #
    # Primary function for image noise mean and sigma calculation
    # Calculates based on pixel values in "flat" image regions
    #
    ### -------- INPUTS -------- ###
    # sun_vec:          Sun vector in image (body) frame
    # img_gray:         Single channel, grayscale image
    # threshold:        Threshold to determine whether region is "flat" or not
    # secondary_threshold:        If no "flat" regions, upper threshold for pixel sample, any pixel value < threshold is sampled
    ### -------- OUTPUTS ----------- ###
    # mean:             Mean of "flat" region or sampled pixels
    # sigma:            Standard deviation of "flat" region or sampled pixels
    ### ------------------------ ###

    top_left_sample = img_gray[0:20, 0:20]
    top_right_sample = img_gray[0:20, -20:]
    bottom_left_sample = img_gray[-20:, 0:20]
    bottom_right_sample = img_gray[-20:, -20:]

    sigma = estimate_sigma(img_gray)

    if sun_vec[0] >= 0 and sun_vec[1] >= 0:
        if sun_vec[0] > sun_vec[1]:
            mean1 = bottom_left_sample.mean()
            if mean1 > threshold:
                mean2 = top_left_sample.mean()
                if mean2 > threshold:
                    mean3 = bottom_right_sample.mean()
                    if mean3 > threshold:
                        mean4 = top_right_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1
        else:
            mean1 = bottom_left_sample.mean()
            if mean1 > threshold:
                mean2 = bottom_right_sample.mean()
                if mean2 > threshold:
                    mean3 = top_left_sample.mean()
                    if mean3 > threshold:
                        mean4 = top_right_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1

    elif sun_vec[0] <= 0 and sun_vec[1] >= 0:
        if abs(sun_vec[0]) < sun_vec[1]:
            mean1 = bottom_right_sample.mean()
            if mean1 > threshold:
                mean2 = bottom_left_sample.mean()
                if mean2 > threshold:
                    mean3 = top_right_sample.mean()
                    if mean3 > threshold:
                        mean4 = top_left_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1
        else:
            mean1 = bottom_right_sample.mean()
            if mean1 > threshold:
                mean2 = top_right_sample.mean()
                if mean2 > threshold:
                    mean3 = bottom_left_sample.mean()
                    if mean3 > threshold:
                        mean4 = top_left_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1

    elif sun_vec[0] <= 0 and sun_vec[1] <= 0:
        if abs(sun_vec[0]) > abs(sun_vec[1]):
            mean1 = top_right_sample.mean()
            if mean1 > threshold:
                mean2 = bottom_right_sample.mean()
                if mean2 > threshold:
                    mean3 = top_left_sample.mean()
                    if mean3 > threshold:
                        mean4 = bottom_left_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1
        else:
            mean1 = top_right_sample.mean()
            if mean1 > threshold:
                mean2 = top_left_sample.mean()
                if mean2 > threshold:
                    mean3 = bottom_right_sample.mean()
                    if mean3 > threshold:
                        mean4 = bottom_left_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1

    elif sun_vec[0] >= 0 and sun_vec[1] <= 0:
        if sun_vec[0] < abs(sun_vec[1]):
            mean1 = top_left_sample.mean()
            if mean1 > threshold:
                mean2 = top_right_sample.mean()
                if mean2 > threshold:
                    mean3 = bottom_left_sample.mean()
                    if mean3 > threshold:
                        mean4 = bottom_right_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1
        else:
            mean1 = top_left_sample.mean()
            if mean1 > threshold:
                mean2 = bottom_left_sample.mean()
                if mean2 > threshold:
                    mean3 = top_right_sample.mean()
                    if mean3 > threshold:
                        mean4 = bottom_right_sample.mean()
                        if mean4 > threshold:
                            mean, sigma = noiseMeanSecondary(
                                img_gray, secondary_threshold)
                        else:
                            mean = mean4
                    else:
                        mean = mean3
                else:
                    mean = mean2
            else:
                mean = mean1
    try:
        mean
    except:
        print('no mean defined')
        print(sun_vec[0])
        print(sun_vec[1])

    return mean, sigma


def gridCoorVal(img_gray, primary_coor):
    #
    # Returns 5x5 grid edge coordinates, shape 3 x m x 25
    # [3:m:0] is input
    # [3:m:1] is the upper left grid (-2,-2 if center is 0,0)
    # [3:m:2] is one pixel down (-2, -1)
    # [3:m:3] is another down (-2, 0)
    # then (-2, 1)
    # (-2, 2)
    # (-1, -2)
    # (-1, -1)
    # ...
    # [3:m:24] is (2, 2)
    #
    ### -------- INPUTS -------- ###
    # img_gray:         Single channel, grayscale image
    # primary_coor:     Primary pixel coordinates
    ### -------- OUTPUTS ----------- ###
    # grid_coors:             3 x m x 25 array with primary and secondary grid pixel coordinates
    # img_values:             n x m x 25 array with primary and secondary grid pixel values
    ### ------------------------ ###

    n, m = primary_coor.shape

    deep_grid_coor = np.zeros((3, m, 24))
    deep_grid_values = np.zeros((1, m, 25))

    img_values = np.zeros((img_gray.shape[0], img_gray.shape[1], 25))

    for m in range(m):
        i, j, one = primary_coor[:, m]
        i, j = int(np.floor(i)), int(np.floor(j))
        deep_grid_values[0, m, 0] = img_gray[j, i]
        x_array = []
        y_array = []
        grid_el = 1
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                else:
                    x_array.append(i+dx)
                    y_array.append(j+dy)
                    if i+dx > img_gray.shape[1]-1 or i+dx < 0 or j+dy > img_gray.shape[0]-1 or j+dy < 0:
                        deep_grid_values[0, m, grid_el] = None
                        grid_el += 1
                    else:
                        deep_grid_values[0, m, grid_el] = img_gray[j+dy, i+dx]
                        grid_el += 1
        x_array = np.asarray(x_array)
        y_array = np.asarray(y_array)
        ones = np.ones(24)
        grid = np.stack((x_array, y_array, ones), axis=0)
        deep_grid_coor[:, m, :] = grid

        img_values[j, i, :] = deep_grid_values[0][m]

    grid_coors = np.concatenate(
        (primary_coor[:, :, np.newaxis], deep_grid_coor), axis=2)

    return grid_coors, img_values


def sobel55(img, R_ib, sun_vec_i, idx_case):
    #
    # Edge detection algorithm
    # 1. Primary scan along lines parallel to sun_vec
    # 2. Sobel run on edge formed by 5x5 grids about primary intercepts
    # 3. Sobel gradient magnitude filtering
    # 4. High-density rescan
    # 5. Edge Cropping
    #
    ### -------- INPUTS -------- ###
    # img:          Source image
    # R_ib:         Single channel, grayscale image
    # sun_vec_i:        Threshold to determine whether region is "flat" or not
    ### -------- OUTPUTS ----------- ###
    # edge_pts:             n x m array with pixel values at edge indices (primary indices)
    # edge_coor:            3 x p edge coordinate array where p is the number of pixels making up the edge (i, j, 1)
    # edge_pts55:           n x m x 25 array with pixel values of primary and secondary grid pixel values
    # edge_coor55:          3 x p x 25 edge_coordnate array with the respective coordinates of each 5x5 grid about each primary pixel
    ### ------------------------ ###

    # Define slope and sun_vec
    sun_vec_b = np.dot(R_ib, sun_vec_i)
    # print('sun_vec_b',sun_vec_b)
    sun_vec_plane = np.hstack([sun_vec_b[0], -sun_vec_b[2]])
    # print('sun_vec_plane',sun_vec_plane)
    slope = sun_vec_plane[1]/-sun_vec_plane[0]  # see functions.py line 990
    # print('slope',slope)
    sun_vec = np.array([-sun_vec_plane[0], sun_vec_plane[1]])
    # print('sun_vec',sun_vec)

    # Convert image to binary using noise_dependent lower binary threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('blackandwhite_image.png',img_gray)
    mean_background, sigma_hat = noiseMean(sun_vec, img_gray, 50)

    lower_binary_threshold = mean_background + 5*sigma_hat
    # print('lower_binary_threshold', lower_binary_threshold)
    ret, img_bin = cv2.threshold(
        img_gray, lower_binary_threshold, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('binary_image.png',img_bin)

    n, m, channels = img.shape
    # if n != 1536 or m != 2048:
    #     print('n and/or m are not expected values (1536, 2048)!')
    #     print('n: ', n)
    #     print('m: ', m)

    # Initialize primary scan
    # Find pixel_start
    pixel_start = offsetLine(n, m, slope)
    drawLine(n, m, slope, pixel_start, 0, 2)

    # Find step size
    step_size_init = 2
    num_parallel_init = 2
    indices, temp = drawLine(n, m, slope, pixel_start,
                             num_parallel_init, step_size_init)
    intercept_indices = scanImage(img_bin, indices, slope, sun_vec)
    step_size = spaceLine(img_bin, intercept_indices, n, m, slope,
                          pixel_start, num_parallel_init, step_size_init, sun_vec)

    # Find num parallel
    num_parallel = numLine(img_bin, n, m, pixel_start,
                           slope, step_size, sun_vec)
    indices, line_mask_image = drawLine(
        n, m, slope, pixel_start, num_parallel, step_size)
    line_mask_inds = indices

    # Find primary intercept inds
    intercept_indices = scanImage(img_bin, indices, slope, sun_vec)
    primary_inds = intercept_indices

    # Define pixel grids about primary intercepts
    mask = np.zeros((n, m))
    grid_mask = generateGrid(intercept_indices)
    grid_mask = np.intc(grid_mask)
    mask[grid_mask[:, 1], grid_mask[:, 0]] = 255

    # Apply mask to original image
    mask = mask.astype('uint8')
    ret, maskb = cv2.threshold(
        mask, lower_binary_threshold, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(img_gray, img_gray, mask=maskb)
    # cv2.imwrite('maskedImage.png', masked_img)

    # Apply sobel to masked image
    grad_x = cv2.Sobel(masked_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(masked_img, cv2.CV_32F, 0, 1, ksize=3)

    # Visualize primary scan lines and intercepts

    # cv2.imwrite('line_mask.png',line_mask_image)
    # primary_indices_array = np.asarray(primary_inds)[:,0]
    # primary_img = np.zeros(img_gray.shape)
    # primary_img[primary_indices_array[:,1],primary_indices_array[:,0]] = 255
    # cv2.imwrite('primary.png',primary_img)


    # g direction filter ### CURRENTLY DEPRECATED
    x_gray = grad_x
    y_gray = grad_y

    gxgy_vec = np.stack((x_gray, y_gray), axis=2)

    ### DEPRECATED ###
    # gxgy_sun_dot = np.dot(gxgy_vec,sun_vec.T)
    # x_gray[gxgy_sun_dot<0] = 0
    # y_gray[gxgy_sun_dot<0] = 0
    # gxgy_vec = np.stack((x_gray,y_gray),axis=2)
    ### DEPRECATED ###

    ### g magnitude filter ###
    g_mag = np.linalg.norm(gxgy_vec, axis=2)
    g_mag_data = g_mag.flatten()
    g_mag_data = g_mag_data[g_mag_data > 0]
    g_mean = g_mag_data.mean()
    g_std = g_mag_data.std()

    # Plot gmag distribution
    """
    fig = plt.figure(figsize =(10, 7)) 
    plt.hist(g_mag_data, bins=25)
    plt.axvline(g_mean, color='r')
    plt.axvline(g_mean+g_std, color='r', ls='--')
    plt.axvline(g_mean+threshold*g_std, color='m')
    plt_title = 'moon_img_' + str(i) + ' gradient magnitude'
    plt.title(plt_title)
    fig_title = 'GmagCases/gmag_hist/moon_img_' + str(i) + '_gmag_hist.png'
    plt.savefig(fig_title)
    plt.close(fig)
    """

    # Apply magnitude filter
    threshold = -0.75
    x_gray[g_mag < g_mean+threshold*g_std] = 0
    y_gray[g_mag < g_mean+threshold*g_std] = 0
    gxgy_vec = np.stack((x_gray, y_gray), axis=2)

    gx_filtered = x_gray
    gy_filtered = y_gray

    # Combine x and y filtered gradients
    gxgy_filtered_combined = np.maximum(gx_filtered, gy_filtered)
    edge_pts = gxgy_filtered_combined

    # Visualize magnitude filtered gradients, x, y, and xy combined
    """
    gx_filename = 'GmagCases/filtered_edges/filtered_' + 'DEBUG' + '_x_edge.png'
    gy_filename = 'GmagCases/filtered_edges/filtered_' + 'DEBUG' + '_y_edge.png'
    cv2.imwrite(gx_filename,gx_filtered)
    cv2.imwrite(gy_filename,gy_filtered)
    
    gxy_filename = 'GmagCases/filtered_edges/filtered_' + 'DEBUG' + '_xy_edge.png'
    cv2.imwrite(gxy_filename, gxgy_filtered_combined)
    """

    # Initial gradient-filter-defined edges
    edge_inds = np.nonzero(edge_pts)
    edge_inds_array = np.transpose(np.array([edge_inds[1], edge_inds[0]]))
    jvec = edge_inds[0]
    ivec = edge_inds[1]
    onevec = np.ones(jvec.shape)
    edge_coor = np.vstack((jvec+0.5, ivec+0.5, onevec))

    edge_pts = np.zeros(x_gray.shape)
    edge_pts[jvec, ivec] = img_gray[jvec, ivec]

    plot_pts = img*0.5
    plot_pts[jvec, ivec, 0] = 255
    # cv2.imwrite('plot_edge.png', plot_pts)
    
    plot_pts = img_bin*0.5
    plot_pts[jvec, ivec] = 255
    # cv2.imwrite('plot_bin_edge.png', plot_pts)

    # High-density rescan
    # Define bounding box
    leftmost_pixel = min(edge_inds_array[:, 0])
    rightmost_pixel = max(edge_inds_array[:, 0])
    topmost_pixel = min(edge_inds_array[:, 1])
    bottommost_pixel = max(edge_inds_array[:, 1])

    bbx = rightmost_pixel-leftmost_pixel
    bby = bottommost_pixel-topmost_pixel

    bounded_edge_img = np.zeros((bby, bbx))
    bounded_edge_img[:, :] = edge_pts[topmost_pixel:bottommost_pixel,
                                      leftmost_pixel:rightmost_pixel]
    ret, bounded_edge_img_bin = cv2.threshold(
        bounded_edge_img, lower_binary_threshold, 255, cv2.THRESH_BINARY)

    # Calibration pixel top left pixel in full res image (0,0 in bounding box)
    calibration_pixel = np.array([leftmost_pixel, topmost_pixel])

    # High density rescan of bounding box
    n_be = bounded_edge_img.shape[0]
    m_be = bounded_edge_img.shape[1]
    pixel_start_be = offsetLine(n_be, m_be, slope)
    rescan_inds_be, line_mask_img_be = drawLine(
        n_be, m_be, slope, pixel_start_be, 3*m_be, 1)

    ### RESCAN DEBUGGING ###
    # Visualize high density scan
    rescan_inds_nonzero = []
    for line in rescan_inds_be:
        if line.shape[0] > 0:
            rescan_inds_nonzero.append(line)

    test_grid = np.zeros(
        (line_mask_img_be.shape[0], line_mask_img_be.shape[1], 3))
    # colors = [[0, 0, 255], [0, 165, 255], [0, 255, 255], [0, 255, 0],
    #           [255, 0, 0], [240, 32, 160], [203, 192, 255], [255, 255, 255]]
    colors = [[125, 125, 125], [0, 0, 0], [0, 0, 0], [0, 0, 0],
              [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    color_idx = 0
    pixel_color = colors[color_idx]
    for line in rescan_inds_nonzero:
        if line.shape[0] > 0:
            for pixel in line:
                ipix = pixel[0]
                jpix = pixel[1]
                test_grid[jpix][ipix] = pixel_color
        color_idx += 1
        pixel_color = colors[color_idx % 7]

    # cv2.imwrite('test_grid.png',test_grid)

    # Find first intercept along each rescan line
    rescan_img_be = np.zeros(edge_pts.shape)
    edge_inds_rescan = []
    for line in rescan_inds_be:

        line = line + calibration_pixel

        # check scan direction for every slope case    
        if slope == 0:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > 0 and slope <= 1:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > 1 and slope < 1000000000000:
            if sun_vec[0] > 0:
                line = line
            else:
                line = np.flipud(line)
        elif slope < -1 and slope > -1000000000000:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        elif slope > -1 and slope < 0:
            if sun_vec[0] > 0:
                line = np.flipud(line)
            else:
                line = line
        else:
            if sun_vec[1] > 0:
                line = line
            else:
                line = np.flipud(line)
        
        for idx_pixel,pixel in enumerate(line):       
            i_be = pixel[0]
            j_be = pixel[1]
            if img_bin[j_be][i_be] > 0:
                # try:
                #     ip_be = line[idx_pixel+4][0]
                #     jp_be = line[idx_pixel+4][1]
                #     edge_inds_rescan.append([ip_be, jp_be])
                #     rescan_img_be[jp_be][ip_be] = 255
                # except:
                #     try:                        
                #         ip_be = line[idx_pixel+3][0]
                #         jp_be = line[idx_pixel+3][1]
                #         edge_inds_rescan.append([ip_be, jp_be])
                #         rescan_img_be[jp_be][ip_be] = 255
                #     except:
                #         try:                        
                #             ip_be = line[idx_pixel+2][0]
                #             jp_be = line[idx_pixel+2][1]
                #             edge_inds_rescan.append([ip_be, jp_be])
                #             rescan_img_be[jp_be][ip_be] = 255
                #         except:                            
                #             try:                        
                #                 ip_be = line[idx_pixel+1][0]
                #                 jp_be = line[idx_pixel+1][1]
                #                 edge_inds_rescan.append([ip_be, jp_be])
                #                 rescan_img_be[jp_be][ip_be] = 255
                #             except:
                edge_inds_rescan.append([i_be, j_be])
                rescan_img_be[j_be][i_be] = 255
                break

    edge_inds_rescan = np.asarray(edge_inds_rescan)

    # Calculate rescan intercept coordinates
    ivec_rescan = edge_inds_rescan[:, 0]
    jvec_rescan = edge_inds_rescan[:, 1]
    onevec_rescan = np.ones(jvec_rescan.shape)
    edge_coor_rescan = np.vstack(
        (jvec_rescan+0.5, ivec_rescan+0.5, onevec_rescan))
        
    plot_pts = img_bin*0.5
    plot_pts[jvec_rescan, ivec_rescan] = 255
    # cv2.imwrite('plot_bin_rescan.png', plot_pts)

    # Calculate rescan intercept values
    edge_pts_rescan = np.zeros(rescan_img_be.shape)
    edge_pts_rescan[jvec_rescan,
                    ivec_rescan] = img_gray[jvec_rescan, ivec_rescan]

    ### RESCAN DEBUGGING ###
    # Visualize rescan lines and intercepts over original image

    test_grid_resize = np.zeros(
        (rescan_img_be.shape[0], rescan_img_be.shape[1], 3))
    start_row = calibration_pixel[1]
    start_col = calibration_pixel[0]
    end_row = start_row + test_grid.shape[0]
    end_col = start_col + test_grid.shape[1]
    test_grid_resize[start_row:end_row, start_col:end_col] = test_grid[:, :]
    # cv2.imwrite('test_grid_resize.png', test_grid_resize)

    img_blend = img.astype(np.uint8)
    test_grid_resize_blend = test_grid_resize.astype(np.uint8)

    rescan_blend = cv2.addWeighted(
        img_blend, 0.5, test_grid_resize_blend, 0.5, 0.0)
    rescan_blend[jvec_rescan, ivec_rescan] = [0, 0, 255]
    # cv2.imwrite('rescan_blend.png', rescan_blend)

    # rescan_edge_inds = np.transpose(np.array([ivec_rescan, jvec_rescan]))    

    # Cropping
    # Filter points closer to terminator
    crop_perc = 0.025  # from each side, 2*crop_perc for total
    vec_len = max(edge_coor_rescan.shape)
    last_pixel = np.round(crop_perc*vec_len)
    if last_pixel==0:
        last_pixel = 1
    jvec_cropped = jvec_rescan[int(np.round(
        crop_perc*vec_len)):-1*int(last_pixel)]
    ivec_cropped = ivec_rescan[int(np.round(
        crop_perc*vec_len)):-1*int(last_pixel)]
    onevec_cropped = np.ones(jvec_cropped.shape)
    edge_coor_cropped = np.vstack(
        (ivec_cropped+0.5, jvec_cropped+0.5, onevec_cropped))

    edge_pts_cropped = np.zeros(rescan_img_be.shape)
    edge_pts_cropped[jvec_cropped,
                     ivec_cropped] = img_gray[jvec_cropped, ivec_cropped]

    # Visualize final edge selection

    # edge_pts_gray_filename = 'GmagCases/edge_pts_gray/edge_pts_gray' + str(i) + '.png'
    # cv2.imwrite(edge_pts_gray_filename,edge_pts_cropped)
    # cv2.imwrite('rescan_cropped.png', edge_pts_cropped)

    ### RESCAN DEBUGGING ###
    # Visualize cropped selection over rescan lines and intercepts over original image

    rescan_blend[jvec_cropped,ivec_cropped] = [255,255,255]
    # rescan_blend_cropped_filename = 'GmagCases/rescan_visualization/rescan' + str(i) + '.png'
    cv2.imwrite('rescan_blend255.png',rescan_blend)    
    # error('s')

    # Visualize selected edge highlighted over original image

    # img_gray_overlay = 0.25*img_gray
    # img_gray_overlay[jvec_cropped,ivec_cropped] = img_gray[jvec_cropped,ivec_cropped]
    # cv2.imwrite('gray_overlay.png',img_gray_overlay)
# 
    # 5x5 output
    edge_coor55_cropped, edge_pts55_cropped = gridCoorVal(
        img_gray, edge_coor_cropped)

    # Cropped Outputs
    edge_pts = edge_pts_cropped
    edge_coor = edge_coor_cropped
    edge_pts55 = edge_pts55_cropped
    edge_coor55 = edge_coor55_cropped

    # Reorder Coordinate Outputs
    # Row-wise, left-to-right, top-to-bottom

    # edge_coor
    edge_coor_i = edge_coor[0, :]
    edge_coor_j = edge_coor[1, :]
    edge_coor_ind = np.lexsort((edge_coor_i, edge_coor_j))

    sorted_edge_coor = np.zeros(edge_coor.shape)
    edge_coor_m = 0
    for el in edge_coor_ind:
        sorted_edge_coor[:, edge_coor_m] = edge_coor[:, el]
        edge_coor_m += 1

    # edge_coor55
    edge_coor55_i = edge_coor55[0, :, 0]
    edge_coor55_j = edge_coor55[1, :, 0]
    edge_coor55_ind = np.lexsort((edge_coor55_i, edge_coor55_j))

    sorted_edge_coor55 = np.zeros(edge_coor55.shape)
    edge_coor55_m = 0
    for el in edge_coor55_ind:
        sorted_edge_coor55[:, edge_coor55_m, :] = edge_coor55[:, el, :]
        edge_coor55_m += 1

    return edge_pts, sorted_edge_coor, edge_pts55, sorted_edge_coor55

# ------------------------------------------------------------------------------


def simulateNoiseColor(img):
    #
    # Simulates 3-channel dark, quantization, and shot noise modeled by
    # respsective distributions and samples from these. Distribution
    # parameters set to match typical small spacecraft camera parameters.
    #
    ### -------- INPUTS -------- ###
    # img:              3-channel numpy array image to add simulated noise to
    # scale_factor:     Multiplies all noise values by given factor
    ### -------- OUTPUTS ----------- ###
    # noisy_img:              Numpy array image, img, with simulated 3-channel noise
    ### ------------------------ ###

    n = img.shape[0]
    m = img.shape[1]
    img_flat = (n, m)

    # define illumniated pixels for SNR calculation
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    illuminated_inds = np.nonzero(img_bin)

    # SNR: signal
    imgrSNR = img[illuminated_inds[0], illuminated_inds[1], 2]
    imggSNR = img[illuminated_inds[0], illuminated_inds[1], 1]
    imgbSNR = img[illuminated_inds[0], illuminated_inds[1], 0]

    # scale 13500 full well capacity to 255 gray level
    # all distribution values from Table 1:
    # https://icubesat.files.wordpress.com/2018/05/b-3-5-201805251613-paper.pdf

    # dark noise
    # modeled by a gaussian distribution with mean of the dark current value
    # and standard deviation of the read noise value
    # 0.1 * 255/13500 = 0.002, 13 * 255/13500 = 0.25
    dnr = np.random.normal(0.002, 0.25, img_flat)
    dng = np.random.normal(0.002, 0.25, img_flat)
    dnb = np.random.normal(0.002, 0.25, img_flat)

    # quantization noise
    # modeled by a uniform distribution
    # 13 * 255/13500 = 0.25, scale by 4 for conservative margin
    qnr = np.random.uniform(-1, 1, img_flat)
    qng = np.random.uniform(-1, 1, img_flat)
    qnb = np.random.uniform(-1, 1, img_flat)

    # shot noise
    # modeled by a poisson distribution
    # lambda = mean of sqrt of illuminated pixel values
    snr = np.random.poisson(np.mean(np.sqrt(imgrSNR)), img_flat)
    sng = np.random.poisson(np.mean(np.sqrt(imggSNR)), img_flat)
    snb = np.random.poisson(np.mean(np.sqrt(imgbSNR)), img_flat)

    # snr = np.random.poisson(2.02, img_flat)
    # sng = np.random.poisson(1.91, img_flat)
    # snb = np.random.poisson(1.57, img_flat)

    # total noise per channel
    tnr = dnr + snr + qnr
    tng = dng + sng + qng
    tnb = dnb + snb + qnb

    # SNR: noise
    tnrSNR = tnr[illuminated_inds]
    tngSNR = tng[illuminated_inds]
    tnbSNR = tnb[illuminated_inds]

    # SNR: signal/noise
    snrr = np.divide(np.mean(imgrSNR), np.mean(np.abs(tnrSNR)))
    snrg = np.divide(np.mean(imggSNR), np.mean(np.abs(tngSNR)))
    snrb = np.divide(np.mean(imgbSNR), np.mean(np.abs(tnbSNR)))

    # scaling
    # rgb SNRs (75, 70, 57)
    r_scale = 75/snrr * 4
    g_scale = 70/snrg * 4
    b_scale = 57/snrb * 4
    # print('r,g,b scale',r_scale,g_scale,b_scale)

    tnr = r_scale*(dnr + qnr) + snr
    tng = g_scale*(dng + qng) + sng
    tnb = b_scale*(dnb + qnb) + snb

    # apply noise to image
    noisy_img = np.zeros(img.shape)
    noisy_img[:, :, 2] = np.rint(img[:, :, 2] + tnr)
    noisy_img[:, :, 1] = np.rint(img[:, :, 1] + tng)
    noisy_img[:, :, 0] = np.rint(img[:, :, 0] + tnb)
    # clip values [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.uint8)

    return noisy_img



def simulateNoiseColor_scaled(img, factor):
    #
    # Simulates 3-channel dark, quantization, and shot noise modeled by
    # respsective distributions and samples from these. Distribution
    # parameters set to match typical small spacecraft camera parameters.
    #
    ### -------- INPUTS -------- ###
    # img:              3-channel numpy array image to add simulated noise to
    # scale_factor:     Multiplies all noise values by given factor
    ### -------- OUTPUTS ----------- ###
    # noisy_img:              Numpy array image, img, with simulated 3-channel noise
    ### ------------------------ ###
    n = img.shape[0]
    m = img.shape[1]
    img_flat = (n, m)
    # define illumniated pixels for SNR calculation
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    ill_inds = np.nonzero(img_bin)
    # SNR: signal
    imgrSNR = img[ill_inds[0], ill_inds[1], 2]
    imggSNR = img[ill_inds[0], ill_inds[1], 1]
    imgbSNR = img[ill_inds[0], ill_inds[1], 0]
    # scale 13500 full well capacity to 255 gray level
    # all distribution values from Table 1:
    # https://icubesat.files.wordpress.com/2018/05/b-3-5-201805251613-paper.pdf
    # dark noise
    # modeled by a gaussian distribution with mean of the dark current value
    # and standard deviation of the read noise value
    # 0.1 * 255/13500 = 0.002, 13 * 255/13500 = 0.25
    dnr = np.random.normal(0.002*factor, 0.25*factor, img_flat)
    dng = np.random.normal(0.002*factor, 0.25*factor, img_flat)
    dnb = np.random.normal(0.002*factor, 0.25*factor, img_flat)
    # quantization noise
    # modeled by a uniform distribution
    # 13 * 255/13500 = 0.25, scale by 4 for conservative margin
    qnr = np.random.uniform(-1*factor, 1*factor, img_flat)
    qng = np.random.uniform(-1*factor, 1*factor, img_flat)
    qnb = np.random.uniform(-1*factor, 1*factor, img_flat)
    # shot noise
    # modeled by a poisson distribution
    # lambda = mean of sqrt of illuminated pixel values
    if imgrSNR.size == 0:
        snr = np.zeros(img_flat)
    else:
        snr = np.random.poisson(np.mean(np.sqrt(imgrSNR)) * factor, img_flat)

    if imggSNR.size == 0:
        sng = np.zeros(img_flat)
    else:
        sng = np.random.poisson(np.mean(np.sqrt(imggSNR)) * factor, img_flat)

    if imgbSNR.size == 0:
        snb = np.zeros(img_flat)
    else:
        snb = np.random.poisson(np.mean(np.sqrt(imgbSNR)) * factor, img_flat)
    # snr = np.random.poisson(2.02, img_flat)
    # sng = np.random.poisson(1.91, img_flat)
    # snb = np.random.poisson(1.57, img_flat)
    # total noise per channel
    tnr = dnr + snr + qnr
    tng = dng + sng + qng
    tnb = dnb + snb + qnb
    # SNR: noise
    if imgrSNR.size == 0 or imggSNR.size == 0 or imgbSNR.size == 0:
        r_scale = 1.0
        g_scale = 1.0
        b_scale = 1.0
    else:
        # SNR: noise
        tnrSNR = tnr[ill_inds]
        tngSNR = tng[ill_inds]
        tnbSNR = tnb[ill_inds]

        # SNR: signal/noise
        snrr = np.divide(np.mean(imgrSNR), np.mean(np.abs(tnrSNR)))
        snrg = np.divide(np.mean(imggSNR), np.mean(np.abs(tngSNR)))
        snrb = np.divide(np.mean(imgbSNR), np.mean(np.abs(tnbSNR)))

        # scaling
        # rgb SNRs (75, 70, 57)
        r_scale = 75 / (snrr / factor)
        g_scale = 70 / (snrg / factor)
        b_scale = 57 / (snrb / factor)

    # print('r,g,b scale',r_scale,g_scale,b_scale)
    tnr = r_scale*(dnr + qnr) + snr
    tng = g_scale*(dng + qng) + sng
    tnb = b_scale*(dnb + qnb) + snb
    # apply noise to image
    noisy_img = np.zeros(img.shape)
    noisy_img[:, :, 2] = np.rint(img[:, :, 2] + tnr)
    noisy_img[:, :, 1] = np.rint(img[:, :, 1] + tng)
    noisy_img[:, :, 0] = np.rint(img[:, :, 0] + tnb)
    # clip values [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.uint8)
    return noisy_img


def simulate_blurred_image(img, k_size=5, sigma_max=3):
    sigma = np.random.uniform(0, sigma_max)
    return cv2.GaussianBlur(img, (k_size, k_size), sigma)


def gg_blur(img,beta_input=[]):
    """
    General gaussian blur func implementation from: 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9079550
    Estimating Generalized Gaussian Blur Kernels for Out-of-Focus Image Deblurring

    Kernel size 25 consistent with ref, beta sampled from
    1 to 1.5 with lower limit enforced by the function and
    the upper limit defined by comparison with argomoon images. 
    """
    def blur_func(x, y, beta):
        # eq. 11 from ref
        alpha = max(2, 1.033*beta - 0.6217)

        return alpha/(2*np.pi*beta**2*gamma(2/alpha)) \
                *np.exp(-(((x**2 + y**2)**0.5)/beta)**alpha)

    def gen_kernel(beta, ksize):
        ker = np.zeros((ksize,ksize))
        for i in range(ksize):
            for j in range(ksize):
                ker[i,j] = blur_func(i-ksize//2, j-ksize//2, beta)
        return ker
    if np.size(beta_input)==0:
        beta = np.random.uniform(1, 1.5) # account for randomness in blur
    else:
        beta = beta_input
    # beta = 1 # account for randomness in blur
    kernel = gen_kernel(beta, 25)
    # img = cv2.imread('moon_img_1.png')
    return cv2.filter2D(img, -1, kernel)

def gg_blur_monte(img,idx_monte):
    """
    General gaussian blur func implementation from: 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9079550

    Kernel size 25 consistent with ref, beta sampled from
    1 to 1.5 with lower limit enforced by the function and
    the upper limit defined by comparison with argomoon images. 
    """
    def blur_func(x, y, beta):
        # eq. 11 from ref
        alpha = max(2, 1.033*beta - 0.6217)

        return alpha/(2*np.pi*beta**2*gamma(2/alpha)) \
                *np.exp(-(((x**2 + y**2)**0.5)/beta)**alpha)

    def gen_kernel(beta, ksize):
        ker = np.zeros((ksize,ksize))
        for i in range(ksize):
            for j in range(ksize):
                ker[i,j] = blur_func(i-ksize//2, j-ksize//2, beta)
        return ker
    
    # beta = np.random.uniform(1, 1.5) # account for randomness in blur
    beta = 1.+0.5*idx_monte/99.
    # beta = 1 # account for randomness in blur
    kernel = gen_kernel(beta, 25)
    # img = cv2.imread('moon_img_1.png')
    return cv2.filter2D(img, -1, kernel)

def moon_ref_to_sun_ref(all_thetaphidist, sun_pos, moon_pos, rot_from_ME_to_J2000):
    theta = all_thetaphidist[:,0]
    phi = all_thetaphidist[:,1]
    dist = all_thetaphidist[:,2]

    sc_xpos = dist*np.cos(theta)*np.cos(phi)
    sc_ypos = dist*np.sin(theta)*np.cos(phi)
    sc_zpos = dist*np.sin(phi)
    sc_pos_wrt_moon = [sc_xpos, sc_ypos, sc_zpos]

    R_sv2i = rot_body2inertial(sun_pos, moon_pos, rot_from_ME_to_J2000)
    R_i2sv = R_sv2i.T
    # sun_vec_sv = np.array([0, -1, 0])
    sc_pos_sv = np.matmul(R_i2sv, sc_pos_wrt_moon)

    sc_pos_sv_xyproj = np.vstack((sc_pos_sv[0,:], sc_pos_sv[1,:], np.zeros((sc_pos_sv.shape[1]))))
    # sun_vec_sv_ext = np.tile(sun_vec_sv, (max(sc_pos_sv.shape), 1)).T
    azimuth = (180/np.pi)*np.arctan2(sc_pos_sv_xyproj[0],
                                    -sc_pos_sv_xyproj[1])
    sc_pos_sv_zproj = sc_pos_sv[2,:]
    inclination = (180/np.pi)*np.arcsin(sc_pos_sv_zproj/np.linalg.norm(sc_pos_sv,axis=0))

    return azimuth, inclination, dist

"""
projects a numpy array with (lon, lat) to (x, y) in mercator coordinates using numpy
license: MIT
adapted from https://github.com/mapbox/mercantile
"""

import numpy as np
import math

def xy_mercator_fun(lng, lat, truncate=False):
    """Convert longitude and latitude to web mercator x, y
    Parameters
    ----------
    lnglat : np.array
        Longitude and latitude array in decimal degrees, shape: (-1, 2)
    truncate : bool, optional
        Whether to truncate or clip inputs to web mercator limits.
    Returns
    -------
    np.array with x, y in webmercator
    >>> a = np.array([(0.0, 0.0), (-75.15963, -14.704620000000013)])
    >>> b = np.array(((0.0, 0.0), (-8366731.739810849, -1655181.9927159143)))
    >>> np.isclose(xy(a), b)
    array([[ True,  True],
           [ True,  True]], dtype=bool)
    """

    # lng, lat = lnglat[:,0], lnglat[:, 1]
    if truncate:
        lng = np.clip(lng, -180.0, 180.0)
        lat = np.clip(lng, -90.0, 90.0)
    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    x = R_Moon * np.radians(lng)
    y = R_Moon * np.log(
        np.tan((math.pi * 0.25) + (0.5 * np.radians(lat))))
    return x,y

def gen_moon_earthMASK(sc_eph, time_eph, camera_definition, iter, ocean_reflection, lookat_input=[], sky_vec_input=[], scenario=[], month_number=[], day=[], hour=[]):

    print('Start of image generation')
    start = timeit.default_timer()

    # determines month of the year -> changes earth map
    year = spice.timout(time_eph, 'YYYY')
    month = spice.timout(time_eph, 'MON')

    ### -------- INPUTS -------- ###
    # sc_eph:       spacecraft coordinates in J2000
    # time_eph:     ephemeris time
    # case:         Scenarios: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    # iter:         Iteration number for image generation purposes
    ### ------------------------ ###

    # Start of time count for the efficiency
    # print('Start of image generation')
    # start = timeit.default_timer()

    # Constants ------------------------------------------------------------------------------------------------------------------------------------------
    # km #scaling factor is: (variable in kilometers)/R_Earth*3.5
    R_Earth = 6378.1
    R_Earth_pole = 6356.8  # km
    highest_pt = 8.850  # km

    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    highest_pt = 11.000         # km

    # Calculations  ------------------------------------------------------------------------------------------------------------------------------------------
    # Position of the Moon compared to Earth in J2000 (ECI) reference frame
    moon = spice.spkpos("301", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    moon_pos_e = [moon[0][0], moon[0][1], moon[0][2]]

    # Position of the Spacecraft in the J2000 frame
    sc_pos = sc_eph

    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]

    # Position of Star Map: centered in Solar System Barycenter (spice ID: 0)
    star = spice.spkpos("0", time_eph, "J2000", "NONE", "399")
    star_pos_e = [star[0][0], star[0][1], star[0][2]]

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", time_eph)

    # Rotation of the Earth on its axis
    rotm_earth = spice.pxform("J2000", "IAU_EARTH", time_eph)

    # Visual Magnitude Parameters
    # Compute vector Earth to Sun
    earth2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    earth2sun = np.array([earth2sun[0][0], earth2sun[0][1], earth2sun[0][2]])
    # Compute the vector Moon to Sun
    moon2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "301")
    moon2sun = np.array([moon2sun[0][0], moon2sun[0][1], moon2sun[0][2]])
    # Compute the vector Earth to Spacecraft
    earth2sc = np.array(sc_pos)
    # Compute the vector Moon to Spacecraft
    moon2sc = np.array(sc_pos) - np.array(moon_pos_e)
    # Call function to calculate the Apparent Magnitude
    V_e, V_m, alpha_m = magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc)
    # print('The apparent magnitude of the Earth is: ', V_e)
    # print('The apparent magnitude of the Moon is: ', V_m)
    # print('The phase angle of the Moon is: '+str("{:.2f}".format(alpha_m))+' deg')

    #Cloud maps: https://www.shadedrelief.com/natural3/pages/clouds.html
    # Texture Maps Earth------------------------------------------------------------------------------------------------------------------------------------------
    if day.size == 0:
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/earth/clouds.jpg") + '"'
        cloud_map = []
    else:       
        if month_number<10:
            str_month = "0" + str(int(month_number))
        else:
            str_month = str(int(month_number))
        if day<10:
            str_day = "0" + str(int(day))
        else:
            str_day = str(int(day))
        if hour<10:
            str_hour = "0" + str(int(hour))
        else:
            str_hour = str(int(hour))
        cloud_map = str(year) + "-" + str_month + "-" + str_day + "T" + str_hour + "-00-00.000Z.jpg"
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/clouds/allyear/" + cloud_map) + '"'
    # color_star_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/TychoSkymapII_v2.tiff") + '"'
    # new_clouds = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/pngkit-cloud-texture-png-961670.tif") + '"'
    # new_land = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_land.tiff") + '"'
    # color_oceanblack = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_oceanblackcontinentwhite.tiff") + '"'

    # color_Earth_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/world_shaded_scaled.jpg") + '"'
    # cloud_fraction = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/cloud_fraction.jpg") + '"'

    # Texture Maps Moon
    # color_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_color_scaled.tiff") + '"'
    # # elevation_fileq = '"' + \
    # #     os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'
    # new_moon = '"' + os.path.join(os.getcwd(), "Maps/moon/new_moon.tiff") + '"'
    # moon_craters2 = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_craters2.tiff") + '"'

    # maps for each month (WITHOUT ocean)
    earth_jan_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jan_adj.jpg") + '"'
    earth_feb_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_feb_adj.jpg") + '"'
    earth_mar_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_mar_adj.jpg") + '"'
    earth_apr_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_apr_adj.jpg") + '"'
    earth_may_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_may_adj.jpg") + '"'
    earth_jun_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jun_adj.jpg") + '"'
    earth_jul_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jul_adj.jpg") + '"'
    earth_aug_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_aug_adj.jpg") + '"'
    earth_sept_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept_adj.jpg") + '"'
    earth_oct_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_oct_adj.jpg") + '"'
    earth_nov_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_nov_adj.jpg") + '"'
    earth_dec_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_dec_adj.jpg") + '"'

    # maps for each month (WITH ocean)
    earth_jan = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jan.jpg") + '"'
    earth_feb = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_feb.jpg") + '"'
    earth_mar = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_mar.jpg") + '"'
    earth_apr = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_apr.jpg") + '"'
    earth_may = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_may.jpg") + '"'
    earth_jun = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jun.jpg") + '"'
    earth_jul = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jul.jpg") + '"'
    earth_aug = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_aug.jpg") + '"'
    earth_sept = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept.jpg") + '"'
    earth_oct = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_oct.jpg") + '"'
    earth_nov = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_nov.jpg") + '"'
    earth_dec = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_dec.jpg") + '"'

    # color_ocean_black_continent_white
    color_oceanblack_jan = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jan.jpg") + '"'
    color_oceanblack_feb = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_feb.jpg") + '"'
    color_oceanblack_mar = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_mar.jpg") + '"'
    color_oceanblack_apr = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_apr.jpg") + '"'
    color_oceanblack_may = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_may.jpg") + '"'
    color_oceanblack_jun = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jun.jpg") + '"'
    color_oceanblack_jul = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jul.jpg") + '"'
    color_oceanblack_aug = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_aug.jpg") + '"'
    color_oceanblack_sept = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_sept.jpg") + '"'
    color_oceanblack_oct = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_oct.jpg") + '"'
    color_oceanblack_nov = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_nov.jpg") + '"'
    color_oceanblack_dec = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_dec.jpg") + '"'

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------

    # determine the month
    if month == 'JAN':
        earth_month = earth_jan
        earth_month_adj = earth_jan_adj
        color_oceanblack_month = color_oceanblack_jan
        earth_map = "earth_jan.jpg"
        earth_adj = "earth_jan_adj.jpg"
        print('JAN')
    elif month == 'FEB':
        earth_month = earth_feb
        earth_month_adj = earth_feb_adj
        color_oceanblack_month = color_oceanblack_feb
        earth_map = "earth_feb.jpg"
        earth_adj = "earth_feb_adj.jpg"
        print('FEB')
    elif month == 'MAR':
        earth_month = earth_mar
        earth_month_adj = earth_mar_adj
        color_oceanblack_month = color_oceanblack_mar
        earth_map = "earth_mar.jpg"
        earth_adj = "earth_mar_adj.jpg"
        print('MAR')
    elif month == 'APR':
        earth_month = earth_apr
        earth_month_adj = earth_apr_adj
        color_oceanblack_month = color_oceanblack_apr
        earth_map = "earth_apr.jpg"
        earth_adj = "earth_apr_adj.jpg"
        print('APR')
    elif month == 'MAY':
        earth_month = earth_may
        earth_month_adj = earth_may_adj
        color_oceanblack_month = color_oceanblack_may
        earth_map = "earth_may.jpg"
        earth_adj = "earth_may_adj.jpg"
        print('MAY')
    elif month == 'JUN':
        earth_month = earth_jun
        earth_month_adj = earth_jun_adj
        color_oceanblack_month = color_oceanblack_jun
        earth_map = "earth_jun.jpg"
        earth_adj = "earth_jun_adj.jpg"
        print('JUN')
    elif month == 'JUL':
        earth_month = earth_jul
        earth_month_adj = earth_jul_adj
        color_oceanblack_month = color_oceanblack_jul
        earth_map = "earth_jul.jpg"
        earth_adj = "earth_jul_adj.jpg"
        print('JUL')
    elif month == 'AUG':
        earth_month = earth_aug
        earth_month_adj = earth_aug_adj
        color_oceanblack_month = color_oceanblack_aug
        earth_map = "earth_aug.jpg"
        earth_adj = "earth_aug_adj.jpg"
        print('AUG')
    elif month == 'SEP':
        earth_month = earth_sept
        earth_month_adj = earth_sept_adj
        color_oceanblack_month = color_oceanblack_sept
        earth_map = "earth_sept.jpg"
        earth_adj = "earth_sept_adj.jpg"
        print('SEP')
    elif month == 'OCT':
        earth_month = earth_oct
        earth_month_adj = earth_oct_adj
        color_oceanblack_month = color_oceanblack_oct
        earth_map = "earth_oct.jpg"
        earth_adj = "earth_oct_adj.jpg"
        print('OCT')
    elif month == 'NOV':
        earth_month = earth_nov
        earth_month_adj = earth_nov_adj
        color_oceanblack_month = color_oceanblack_nov
        earth_map = "earth_nov.jpg"
        earth_adj = "earth_nov_adj.jpg"
        print('NOV')
    elif month == 'DEC':
        earth_month = earth_dec
        earth_month_adj = earth_dec_adj
        color_oceanblack_month = color_oceanblack_dec
        earth_map = "earth_dec.jpg"
        earth_adj = "earth_dec_adj.jpg"
        print('DEC')

# Other Objects of Scene()
    if scenario==[]:
        pov_name = "tmp_" + str(int(iter)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '.ini'
    else:
        pov_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.ini'
    pov_file = open(pov_namesingle,
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # pov_file.write('// GLOBAL SETTINGS\n'
    #         'global_settings { \n'
    #         '    assumed_gamma 1.0\n'
    #         '    radiosity { \n'
    #         '        count 1000\n'
    #         '        nearest_count 20\n'
    #         '        recursion_limit 1\n'
    #         '        normal on\n'
    #         '    }\n'
    #         '}\n')

    # ocean reflection on = 1, off = 0, original maps = 2
    if ocean_reflection == 0:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'diffuse', 0.85, 'brilliance', 1.0)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 diffuse 0.85 brilliance 1.0}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 1: #new version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'reflection', "{0.05,0.25}", 'diffuse', 0.85, 'brilliance', 1.0, 'phong', 0.06, 'phong_size', 30)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            # '                   [0.000 pigment{color rgb <0.007, 0.03, 0.1>*0.6} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'
            '                   [0.000 pigment{color rgb <0.0, 0.0, 0.0>} finish{ambient 0.0 reflection {0.0} diffuse 1 brilliance 0 phong 0.0}]\n'
            '                   [1.000 pigment{color rgb <1.00, 1.0, 1.0>} finish{ambient 0.0 reflection {0.0} diffuse 1 brilliance 0 phong 0.0}]\n'


            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 2: #old version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(Pigment(ImageMap('jpeg', earth_month, "map_type", 1)), Finish(
        #                       'ambient', 0.0)),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           pigment{\n'
            '               image_map{\n'
            '                   jpeg ' + earth_month + ' map_type 1\n'
            '               }\n'
            '           }\n'
            '           finish{ambient 0.0}\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    # color_oceanblack: ocean black, continents white.
    # TextureMap 0.00: set color of the ocean.
    # TextureMap 1.00: use earth map for continents

    # atmos = Difference(Sphere([0, 0, 0], 3.504, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                    Sphere([0, 0, 0], 3.501, 'scale', [
    #                           1., R_Earth_pole/R_Earth, 1.]),
    #                    Material(Texture(Pigment("rgbt 1")), Interior(Media(Scattering(5, "color White", "eccentricity 0.56"),
    #                                                                        Density('spherical', DensityMap([0.0, "rgb 0.0"], [0.5294*0.25e-6, "rgb <0.02, 0.05, 0.2>*0.07"],
    #                                                                                                        [0.5294*0.4e-6, "rgb <0.02, 0.07, 0.3>*0.32"], [
    #                                                                            0.5294*0.5e-6,   "rgb <0.08, 0.18, 0.4>*0.5"],
    #                                                                            [0.5412*0.6e-6, "rgb <0.08, 0.18, 0.4>*0.9"], [
    #                                                                            0.5471*0.65e-6,  "rgb <0.08, 0.18, 0.4>*1.5"],
    #                                                                            [0.5471*0.675e-6, "rgb <0.08, 0.18, 0.4>*4.5"], [0.5471*0.71e-6,  "rgb <0.08, 0.18, 0.4>*12"]), "scale 3.52")))),
    #                    'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z')

    # pov_file.write(
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5003\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5001\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt 1\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     '                   scattering{\n'
    #     '                       4 color White\n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     '                           [0.0 rgb 0.0]\n'
    #     '                           [0.5294*0.25e-6 rgb <0.4, 0.5, 0.6>*0.07*10]\n'
    #     '                           [0.5294*0.4e-6 rgb <0.4, 0.5, 0.6>*0.32*10]\n'
    #     '                           [0.5294*0.5e-6 rgb <0.4, 0.5, 0.6>*0.5*10]\n'
    #     '                           [0.5412*0.6e-6 rgb <0.4, 0.5, 0.6>*0.9*10]\n'
    #     '                           [0.5471*0.65e-6 rgb <0.4, 0.5, 0.6>*1.5*10]\n'
    #     '                           [0.5471*0.685e-6 rgb <0.4, 0.5, 0.6>*4.5*10]\n'
    #     '                           [0.5471*0.69e-6 rgb <0.4, 0.6, 0.8>*4*10]\n'
    #     '                       }\n'
    #     '                       scale 3.5003\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 1\n'
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 0.989\n' #0.989
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt <1,1,1,1>\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
    #     '                   scattering { 4 rgb <0.4, 0.6, 0.8> \n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
    #     # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
    #     # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
    #     # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
    #     # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0 rgb <1,1,1>*0.]\n'
    #     # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
    #     # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
    #     # '                           [0.0039 rgb <1,1,1>*0.000]\n'
    #     '                           [0.0063 rgbt 0.0]\n'
    #     '                           [0.0085 rgbt 0.1]\n'
    #     # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
    #     '                           [0.0111 rgbt 1.0]\n'
    #     '                           [0.0111 rgbt 0.0]\n'
    #     '                           [1.0 rgbt 0.0]\n'
    #     '                       }\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
    #     '       scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'
    #     # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
    #     '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
    #     '#declare fBaseRadius = '+str(3.5)+';\n'
    #     '#declare fAtmoFactor = '+str(1.0114)+';\n'
    #     '#declare cAtmoColor = <1,1,1>;'
    #     '#declare M_Atmosphere_Intervals = 10;\n'
    #     '#declare M_Atmosphere_Samples = 10;\n'
    #     '#declare ATMO_STEPS = 9;\n'
    #     '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare M_Atmosphere =\n'
    #     '   media { \n'
    #     # '	    emission 0.3\n'
    #     '       scattering { 4 color rgb <1,1,1> \n'
    #     '	    intervals M_Atmosphere_Intervals\n'
    #     '	    samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
    #     '	    method 3\n'
    #     '	    density { spherical\n'
    #     '		    color_map {\n'
    #     '		        [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
    #     '               #local I=1;#while(I<ATMO_STEPS)\n'
    #     '               #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
    #     # '               #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
    #     '               #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
    #     # '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
    #     '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
    #     '               #local I=I+1;#end\n'
    #     # '			    [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '               [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
    #     '			    [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '			    [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
    #     '           }\n'
    #     '       }\n'
    #     '   }\n'
    #     '#declare atmos = \n'
    #     '   sphere { < 0, 0, 0>, 1\n'
    #     '       no_shadow\n'
    #     '       hollow off\n'        
    #     '       texture { pigment { P_EmittingAtmo } } // sun-side emission\n'        
    #     '       no_shadow\n'
    #     '       hollow on\n'        
    #     '       interior { media { M_Atmosphere } }\n'        
    #     '       finish { surface_finish }\n'
    #     '       scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    #https://github.com/pyramid3d/space3d/blob/master/include/s3d_scene.inc
    pov_file.write(
        # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
        '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
        '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
        '#declare fBaseRadius = '+str(3.5)+';\n'
        '#declare fAtmoFactor = '+str(1.0114)+';\n'
        '#declare cAtmoColor = <1,1,1>;'
        '#declare M_Atmosphere_Intervals = 1;\n'
        '#declare M_Atmosphere_Samples = 1;\n'
        '#declare ATMO_STEPS = 9;\n'
        '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_STRENGTH = 20;\n'
        '#declare ATMO_STRENGTH = 25;\n'
        # '#declare M_Atmosphere =\n'
        '#declare atmos = \n'
        # '   difference {\n'
        '   sphere {\n'
        '       <0, 0, 0>, 1\n'
        # '       scale <1., 1., 1.>\n'
        # '   }\n'
        # '       sphere {\n'
        # '           <0, 0, 0>, 0.989\n' #0.989
        # '           scale <1., 1., 1.>\n'
        # '       }\n'
        '       material{\n'
        '           texture{\n'
        '               pigment{\n'
        '                   rgbt <1,1,1,1>\n'
        '               }\n'
        '           }\n'
        '           interior{\n'
        '           media{\n'
    # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
        # '               scattering { 4 rgb <0.4, 0.6, 0.8> \n'
        '               scattering { 4 rgb <0.3, 0.4, 0.5> }\n'
        '	            intervals M_Atmosphere_Intervals\n'
        '	            samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
        '               density{\n'
        '                   spherical\n'
        '                   density_map {\n'
        '                       [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
        '                       #local I=1;#while(I<ATMO_STEPS)\n'
        '                       #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
        # '                           #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
        '                       #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
        # '                           [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
        '                       [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
        '                       #local I=I+1;#end\n'
        # '			                [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
        '                   }\n'
        # '                       density_map{\n'
        # # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
        # # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
        # # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
        # # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
        # # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0 rgb <1,1,1>*0.]\n'
        # # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
        # # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
        # # '                           [0.0039 rgb <1,1,1>*0.000]\n'
        # '                           [0.0063 rgbt 0.0]\n'
        # '                           [0.0085 rgbt 0.1]\n'
        # # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
        # '                           [0.0111 rgbt 1.0]\n'
        # '                           [0.0111 rgbt 0.0]\n'
        # '                           [1.0 rgbt 0.0]\n'
        # '                       }\n'
        '                       }\n'
        '                   }\n'
        '               }\n'
        '           }\n'
        # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
        # '           scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'        
        '           scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
        # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
        '           hollow on\n'        
        '           no_shadow\n'
        '           finish { surface_finish }\n'
        '           rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '           matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

        # '                           [1.0 rgb 0.]\n'
        # '                           [0.0109 rgb 0.]\n'
        # '                           [0.0109 rgb 1.]\n'
        # '                           [0.0103 rgb 0.5]\n'
        # '                           [0.0085 rgb 0.1]\n'
        # '                           [0.0063 rgb 0.01]\n'
        # '                           [0.0039 rgb 0.001]\n'
        # '                           [0.0009 rgb 0.0001]\n'
        # '                           [0.0 rgb 0.0]\n'
    
        # '                           [1.0 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.4, 0.6, 0.8>*10000]\n'
        # '                           [0.0103 rgb <0.4*0.5, 0.6*0.5, 0.8*0.5>*10000]\n'
        # '                           [0.0085 rgb <0.4*0.1, 0.6*0.1, 0.8*0.1>*10000]\n'
        # '                           [0.0063 rgb <0.4*0.01, 0.6*0.01, 0.8*0.01>*10000]\n'
        # '                           [0.0039 rgb <0.4*0.001, 0.6*0.001, 0.8*0.001>*10000]\n'
        # '                           [0.0009 rgb <0.4*0.0001, 0.6*0.0001, 0.8*0.0001>*10000]\n'
        # '                           [0.0 rgbt <0,0,0,1>*1000]\n'

    # clouds = Difference(Sphere([0, 0, 0], 3.502, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                     Sphere([0, 0, 0], 3.501, 'scale', [
    #                            1., R_Earth_pole/R_Earth, 1.]),
    #                     Texture(Pigment(ImagePattern('tiff', color_clouds_fileq, "map_type", 1),
    #                                     ColorMap([0.03, 'color', "rgbt <1,1,1,1>"], [1.0, 'color', "rgbt <1,1,1,0>"])),
    #                             Finish("ambient 0.0")), 'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                     'matrix', '<' + str(rotm_earth[0][0]) + ',' +
    #                     str(rotm_earth[0][1]) + ',' +
    #                     str(rotm_earth[0][2]) + ','
    #                     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #                     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>')
    # alternate cloud map
    #     clouds =    Difference(Sphere([0,0,0],3.502, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Sphere([0,0,0],3.501, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Texture(Pigment(ImagePattern('jpeg', cloud_fraction, "map_type", 1),
    #                 ColorMap([0.66, 'color', "rgbt <1,1,1,1>"],[0.95, 'color', "rgbt <1,1,1,0>"])),
    #                 Finish("ambient 0.0")),'hollow on','rotate','-90*x','scale','<1,1,-1>','rotate','180*z',
    #                 'matrix', '<' + str(rotm_earth[0][0]) + ',' + str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #                 + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    #                 + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')

    #0.03
    pov_file.write(
        '#declare clouds = \n'
        '   difference {\n'
        '       sphere {\n'
        '           <0, 0, 0>, 3.5004\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       sphere {\n'
        # '           <0, 0, 0>, 3.5001\n'
        '           <0, 0, 0>, 3.5002\n'
        '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
        '       }\n'
        '       texture {\n'
        '           pigment{\n'
        '               image_pattern{\n'
        '                   jpeg ' + color_clouds_fileq + ' map_type 1\n'
        '               }\n'
        '               color_map{\n'
        # '                   [0.15 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        # '                   [0.0 color rgbt <0.7,0.7,0.7,1>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [0.2 color rgbt <0.7,0.7,0.7,0.975>]\n'
        # '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'        
        '                   [0.0 color rgbt <0.8,0.8,0.8,1>]\n'
        '                   [0.3 color rgbt <0.7,0.7,0.7,0.99>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [0.4 color rgbt <0.7,0.7,0.7,0.9>]\n'
        '                   [1.0 color rgbt <0.8,0.8,0.8,0.2>]\n'
        '               }\n'
        '           }\n'
        '           finish{ambient 0.0}\n'
        '       }\n'
        '       hollow on\n'     
        '       no_shadow\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # ------------------------------------------------------------------------------------------------------
    # Moon Sphere
    # scaling factor for the Earth Moon environment (Earth primary central body)
    scale_m = R_Moon/R_Earth*3.5
    # imported values range from 0 to 1, transform to -1 to 1, then from -32768 to 32767, in this range, multiply by factor 0.5 to convert into meters
    # ele = 'ele=function{pigment{image_map{tiff ".Maps/moon/elevation_20.tiff" map_type 1 interpolate 2}}}'

 # Accuracy Parameter
    sc_dist = np.array(sc_pos) - np.array(moon_pos_e)
    dist = norm(sc_dist)    # in km
    if dist <= 3000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_1.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 10000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_2.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 20000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 30000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 70000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 100000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    else:
        acc = 0.001     
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
        
    pov_file.write(
    '#declare ele = \n'
    '   function{\n'
    '       pigment{\n'
    '           image_map{\n'
    '               tiff ' + elevation_fileq + ' map_type 1 interpolate 2\n'
    '           }\n'
    '       }\n'
    '   }\n')
    

#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture(Pigment(ImageMap('tiff', moon_darklight, "map_type", 1)),
#                 Finish('ambient', "0", 'diffuse', 0.7, 'brilliance', 1.5,)), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
        # The rotation of the sphere is done by using the matrix parameter (https://www.povray.org/documentation/view/3.6.0/49/) which takes directly the elements of the rotation matrix to compute the rotation. The rotation matrix is obtained through spice and it is from inertial (J2000) to body fixed (IAU_MOON)
        # adjust properties so we can use a single light source
#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture( ImagePattern( 'tiff', moon_craters2, "map_type", 1),
#                     TextureMap([0.000,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.8,'brilliance',1.2)],
#                                [0.001,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                [0.999,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                 [1.000, Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                                  Finish('ambient',0.0,'diffuse',0.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
    # sphere_m = Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
    #                       ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)
    #                                   ), 'accuracy', str(acc), 'max_gradient', 1.2,  # str(acc)
    #                       Texture(ImagePattern('tiff', moon_adj, "map_type", 1),
    #                               TextureMap([0.0, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.2, 'brilliance', 1.2)],
    #                                          [0.2, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.25)],
    #                                          [0.5, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.4)],
    #                                          [1.000, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.9, 'brilliance', 1.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                       'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
    #                        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
    #                                                          [1]) + ',' + str(rotm_moon[1][2]) + ','
    #                       + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
    #                                                          ) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
    #                       'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')

    pov_file.write(
        '#declare sphere_m = \n'
        '   isosurface {\n'
        '       function {\n'
        '           f_sphere(x,y,z,' + str(scale_m) +
        ') -  (ele(x,y,z).gray-0.5)*(2*32.767/2/' + str(R_Earth/3.5) + ')\n'
        '           }\n'
        '       contained_by {\n'
        '           sphere {0, (1 + ' + str(highest_pt /
                                            R_Moon) + ')*' + str(scale_m) + '}\n'
        '           }\n'
        '       accuracy ' + str(acc) + ' max_gradient 2.0\n'
        '       texture {\n'
        '           image_pattern{\n'
        '               tiff ' + moon_adj + ' map_type 1 interpolate 2\n'
        '           }\n'
        '           texture_map{\n'
        '               [0.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.2 brilliance 1.2}]\n'
        '               [0.2 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.25}]\n'
        '               [0.5 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.4}]\n'
        '               [1.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.9 brilliance 1.2}]\n'
        '           }\n'
        '       }\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_moon[0][0]) + ',' +
        str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
                                           [1]) + ',' + str(rotm_moon[1][2]) + ','
        + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
                                           ) + ',' + str(rotm_moon[2][2]) + ', 0, 0, 0>\n'
        '       translate <' + str(moon_pos_e[0]/R_Earth*3.5) + ',' + str(
            moon_pos_e[1]/R_Earth*3.5) + ',' + str(moon_pos_e[2]/R_Earth*3.5) + '>\n'
        '   }\n')

    # adjust regions -> 0,0.4,0.8,1.0
    # no need for moon_Craters2, just use moon_adj

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # scene = Scene(camera, objects=[light, sphere_e, clouds, sphere_m, atmos], included=[
    #     "functions.inc", "colors.inc"], declares=[ele])

    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    if look_at_object == 'EARTH':
        pov_file.write('object {sphere_e}\n')
        # pov_file.write('object {atmos}\n')
        # pov_file.write('object {clouds}\n')
    elif look_at_object == 'MOON':
        pov_file.write('object {sphere_m}\n')

    if look_at_object == 'EARTH':
        print('EARTH IMAGE')
        if scenario==[]:
            image_name = "earth_img_MASK" + str(int(iter)) + ".png"
        else:
            image_name = "earth_img_MASK" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            look_at = [0, 0, 0]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]
        
    elif look_at_object == 'MOON':
        print('MOON IMAGE')
        if scenario==[]:
            image_name = "moon_img_" + str(int(iter)) + ".png"
        else:
            image_name = "moon_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            # if dist-R_Moon<= 5000:
            #     sc_pos_rel_moon =  sc_pos-moon_pos_e
            #     norm_pos_rel_moon = np.linalg.norm(sc_pos_rel_moon)
            #     look_at = [moon_pos_e[0]/R_Earth*3.5 + sc_pos_rel_moon[0]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[1] /
            #             R_Earth*3.5 + sc_pos_rel_moon[1]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5 + sc_pos_rel_moon[2]/norm_pos_rel_moon*R_Moon/R_Earth*3.5]
            # else:
            look_at = [moon_pos_e[0]/R_Earth*3.5, moon_pos_e[1] /
                    R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]

    scene_file = os.path.join(os.getcwd(), image_name)
    scene_fileq = '"' + scene_file + '"'

    # camera = Camera('right', '<-' + str(cam_width/cam_height) + ',0,0>', 'sky', 'z', 'direction', 'z', 'angle', str(fov_angle),
    #                 'location', [sc_pos[0]/R_Earth*3.5, sc_pos[1] /
    #                              R_Earth*3.5, sc_pos[2]/R_Earth*3.5], 'up', 'z',
    #                 'look_at', look_at)

    # Camera with FOV and Resolution Implementation
    if len(sky_vec_input)==0:    
        sc2obj = np.array(look_at)-np.array(sc_pos)/R_Earth*3.5
        sc2obj = sc2obj/np.linalg.norm(sc2obj)
        # print(sc2obj)
        if norm(np.cross(sc2obj, np.array((0,0,1))))<1e-14:            
            x_b = np.array((1,0,0))           
            z_b = np.cross(x_b, sc2obj)
            sky_vec = '<' + str(z_b[0]) + ',' + str(z_b[1]) + ',' + str(z_b[2]) + '>'
            # print(sky_vec)
            # error('s')
        else:
            sky_vec = 'z'
        # Axis z the vertical of the camera plane
    else:
        sky_vec = '<' + str(sky_vec_input[0]) + ',' + str(sky_vec_input[1]) + ',' + str(sky_vec_input[2]) + '>'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    up z\n'
                   '    direction z\n'
                   '    sky ' + sky_vec + '\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' + str(sc_pos[0]/R_Earth*3.5) + ',' + str(
                       sc_pos[1]/R_Earth*3.5) + ',' + str(sc_pos[2]/R_Earth*3.5) + '>\n'
                   '    look_at <' +
                   str(look_at[0]) + ',' + str(look_at[1]) +
                   ',' + str(look_at[2]) + '>\n'
                   '}\n')

    # light source
    # light = LightSource([sun_pos_e[0]/R_Earth*3.5, sun_pos_e[1] /
    #                     R_Earth*3.5, sun_pos_e[2]/R_Earth*3.5], 'color', "rgb 2.4")

    pov_file.write('light_source { \n'
                   '    <' + str(sun_pos_e[0]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[1]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[2]/R_Earth*3.5) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    # width and height define the resolution of the image generated
    # scene.render(scene_file, width=cam_width, height=cam_height)

    ini_file = open(ini_namesingle,
                    'w')  # .ini temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        'Antialias=On\n'
        'Antialias_Threshold=0.0\n'
        'Antialias_Depth=3\n'
        # 'Max_Image_Buffer_Memory=' + str(100000) + '\n'
        'Input_File_Name=' + '"' +
        os.path.join(os.getcwd(), pov_name) + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(os.getcwd(), ini_name) + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        print(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')
        os.system(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')

    stop = timeit.default_timer()
    print('End - Total Time: ' + str("{:.2f}".format(stop - start)) + ' s')

# OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(color_src)
    # plt.axis('off')
    # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()
# end render------------------------------------------------
    return scene_file, earth_map, cloud_map

def gen_moon_earthCLOUDMASK(sc_eph, time_eph, camera_definition, iter, ocean_reflection, lookat_input=[], sky_vec_input=[], scenario=[], month_number=[], day=[], hour=[]):

    print('Start of image generation')
    start = timeit.default_timer()

    # determines month of the year -> changes earth map
    year = spice.timout(time_eph, 'YYYY')
    month = spice.timout(time_eph, 'MON')

    ### -------- INPUTS -------- ###
    # sc_eph:       spacecraft coordinates in J2000
    # time_eph:     ephemeris time
    # case:         Scenarios: 1 = around Moon looking at Moon, 2 = around Earth looking at Moon, 3 = around Earth looking at Earth
    # iter:         Iteration number for image generation purposes
    ### ------------------------ ###

    # Start of time count for the efficiency
    # print('Start of image generation')
    # start = timeit.default_timer()

    # Constants ------------------------------------------------------------------------------------------------------------------------------------------
    # km #scaling factor is: (variable in kilometers)/R_Earth*3.5
    R_Earth = 6378.1
    R_Earth_pole = 6356.8  # km
    highest_pt = 8.850  # km

    # km, scaling factor is: (variable in km)/R_Moon
    R_Moon = 1737.400  # https://svs.gsfc.nasa.gov/cgi-bin/details.cgi?aid=4720
    highest_pt = 11.000         # km

    # Calculations  ------------------------------------------------------------------------------------------------------------------------------------------
    # Position of the Moon compared to Earth in J2000 (ECI) reference frame
    moon = spice.spkpos("301", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    moon_pos_e = [moon[0][0], moon[0][1], moon[0][2]]

    # Position of the Spacecraft in the J2000 frame
    sc_pos = sc_eph

    # Position of the Sun compared to Earth in J2000 reference frame
    sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    # To extract the position array
    sun_pos_e = [sun[0][0], sun[0][1], sun[0][2]]

    # Position of Star Map: centered in Solar System Barycenter (spice ID: 0)
    star = spice.spkpos("0", time_eph, "J2000", "NONE", "399")
    star_pos_e = [star[0][0], star[0][1], star[0][2]]

    # Rotation of the Moon on its axis
    rotm_moon = spice.pxform("J2000", "MOON_ME", time_eph)

    # Rotation of the Earth on its axis
    rotm_earth = spice.pxform("J2000", "IAU_EARTH", time_eph)

    # Visual Magnitude Parameters
    # Compute vector Earth to Sun
    earth2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "399")
    earth2sun = np.array([earth2sun[0][0], earth2sun[0][1], earth2sun[0][2]])
    # Compute the vector Moon to Sun
    moon2sun = spice.spkpos("10", time_eph, "J2000", "NONE", "301")
    moon2sun = np.array([moon2sun[0][0], moon2sun[0][1], moon2sun[0][2]])
    # Compute the vector Earth to Spacecraft
    earth2sc = np.array(sc_pos)
    # Compute the vector Moon to Spacecraft
    moon2sc = np.array(sc_pos) - np.array(moon_pos_e)
    # Call function to calculate the Apparent Magnitude
    V_e, V_m, alpha_m = magnitude_calc(earth2sun, moon2sun, earth2sc, moon2sc)
    # print('The apparent magnitude of the Earth is: ', V_e)
    # print('The apparent magnitude of the Moon is: ', V_m)
    # print('The phase angle of the Moon is: '+str("{:.2f}".format(alpha_m))+' deg')

    #Cloud maps: https://www.shadedrelief.com/natural3/pages/clouds.html
    # Texture Maps Earth------------------------------------------------------------------------------------------------------------------------------------------
    if day.size == 0:
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/earth/clouds.jpg") + '"'
        cloud_map = []
    else:       
        if month_number<10:
            str_month = "0" + str(int(month_number))
        else:
            str_month = str(int(month_number))
        if day<10:
            str_day = "0" + str(int(day))
        else:
            str_day = str(int(day))
        if hour<10:
            str_hour = "0" + str(int(hour))
        else:
            str_hour = str(int(hour))
        cloud_map = str(year) + "-" + str_month + "-" + str_day + "T" + str_hour + "-00-00.000Z.jpg"
        color_clouds_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/clouds/allyear/" + cloud_map) + '"'
    # color_star_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/TychoSkymapII_v2.tiff") + '"'
    # new_clouds = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/pngkit-cloud-texture-png-961670.tif") + '"'
    # new_land = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_land.tiff") + '"'
    # color_oceanblack = '"' + \
    #     os.path.join(
    #         os.getcwd(), "Maps/earth/world_shaded_scaled_oceanblackcontinentwhite.tiff") + '"'

    # color_Earth_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/world_shaded_scaled.jpg") + '"'
    # cloud_fraction = '"' + \
    #     os.path.join(os.getcwd(), "Maps/earth/cloud_fraction.jpg") + '"'

    # Texture Maps Moon
    # color_fileq = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_color_scaled.tiff") + '"'
    # # elevation_fileq = '"' + \
    # #     os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'
    # new_moon = '"' + os.path.join(os.getcwd(), "Maps/moon/new_moon.tiff") + '"'
    # moon_craters2 = '"' + \
    #     os.path.join(os.getcwd(), "Maps/moon/moon_craters2.tiff") + '"'

    # maps for each month (WITHOUT ocean)
    earth_jan_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jan_adj.jpg") + '"'
    earth_feb_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_feb_adj.jpg") + '"'
    earth_mar_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_mar_adj.jpg") + '"'
    earth_apr_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_apr_adj.jpg") + '"'
    earth_may_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_may_adj.jpg") + '"'
    earth_jun_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jun_adj.jpg") + '"'
    earth_jul_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_jul_adj.jpg") + '"'
    earth_aug_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_aug_adj.jpg") + '"'
    earth_sept_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept_adj.jpg") + '"'
    earth_oct_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_oct_adj.jpg") + '"'
    earth_nov_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_nov_adj.jpg") + '"'
    earth_dec_adj = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_dec_adj.jpg") + '"'

    # maps for each month (WITH ocean)
    earth_jan = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jan.jpg") + '"'
    earth_feb = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_feb.jpg") + '"'
    earth_mar = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_mar.jpg") + '"'
    earth_apr = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_apr.jpg") + '"'
    earth_may = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_may.jpg") + '"'
    earth_jun = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jun.jpg") + '"'
    earth_jul = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_jul.jpg") + '"'
    earth_aug = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_aug.jpg") + '"'
    earth_sept = '"' + \
        os.path.join(os.getcwd(), "Maps/months/earth_sept.jpg") + '"'
    earth_oct = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_oct.jpg") + '"'
    earth_nov = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_nov.jpg") + '"'
    earth_dec = '"' + os.path.join(os.getcwd(),
                                   "Maps/months/earth_dec.jpg") + '"'

    # color_ocean_black_continent_white
    color_oceanblack_jan = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jan.jpg") + '"'
    color_oceanblack_feb = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_feb.jpg") + '"'
    color_oceanblack_mar = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_mar.jpg") + '"'
    color_oceanblack_apr = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_apr.jpg") + '"'
    color_oceanblack_may = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_may.jpg") + '"'
    color_oceanblack_jun = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jun.jpg") + '"'
    color_oceanblack_jul = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_jul.jpg") + '"'
    color_oceanblack_aug = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_aug.jpg") + '"'
    color_oceanblack_sept = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_sept.jpg") + '"'
    color_oceanblack_oct = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_oct.jpg") + '"'
    color_oceanblack_nov = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_nov.jpg") + '"'
    color_oceanblack_dec = '"' + \
        os.path.join(
            os.getcwd(), "Maps/color_ocean/color_oceanblack_dec.jpg") + '"'

    # Scene Generation  ------------------------------------------------------------------------------------------------------------------------------------------

    # determine the month
    if month == 'JAN':
        earth_month = earth_jan
        earth_month_adj = earth_jan_adj
        color_oceanblack_month = color_oceanblack_jan
        earth_map = "earth_jan.jpg"
        earth_adj = "earth_jan_adj.jpg"
        print('JAN')
    elif month == 'FEB':
        earth_month = earth_feb
        earth_month_adj = earth_feb_adj
        color_oceanblack_month = color_oceanblack_feb
        earth_map = "earth_feb.jpg"
        earth_adj = "earth_feb_adj.jpg"
        print('FEB')
    elif month == 'MAR':
        earth_month = earth_mar
        earth_month_adj = earth_mar_adj
        color_oceanblack_month = color_oceanblack_mar
        earth_map = "earth_mar.jpg"
        earth_adj = "earth_mar_adj.jpg"
        print('MAR')
    elif month == 'APR':
        earth_month = earth_apr
        earth_month_adj = earth_apr_adj
        color_oceanblack_month = color_oceanblack_apr
        earth_map = "earth_apr.jpg"
        earth_adj = "earth_apr_adj.jpg"
        print('APR')
    elif month == 'MAY':
        earth_month = earth_may
        earth_month_adj = earth_may_adj
        color_oceanblack_month = color_oceanblack_may
        earth_map = "earth_may.jpg"
        earth_adj = "earth_may_adj.jpg"
        print('MAY')
    elif month == 'JUN':
        earth_month = earth_jun
        earth_month_adj = earth_jun_adj
        color_oceanblack_month = color_oceanblack_jun
        earth_map = "earth_jun.jpg"
        earth_adj = "earth_jun_adj.jpg"
        print('JUN')
    elif month == 'JUL':
        earth_month = earth_jul
        earth_month_adj = earth_jul_adj
        color_oceanblack_month = color_oceanblack_jul
        earth_map = "earth_jul.jpg"
        earth_adj = "earth_jul_adj.jpg"
        print('JUL')
    elif month == 'AUG':
        earth_month = earth_aug
        earth_month_adj = earth_aug_adj
        color_oceanblack_month = color_oceanblack_aug
        earth_map = "earth_aug.jpg"
        earth_adj = "earth_aug_adj.jpg"
        print('AUG')
    elif month == 'SEP':
        earth_month = earth_sept
        earth_month_adj = earth_sept_adj
        color_oceanblack_month = color_oceanblack_sept
        earth_map = "earth_sept.jpg"
        earth_adj = "earth_sept_adj.jpg"
        print('SEP')
    elif month == 'OCT':
        earth_month = earth_oct
        earth_month_adj = earth_oct_adj
        color_oceanblack_month = color_oceanblack_oct
        earth_map = "earth_oct.jpg"
        earth_adj = "earth_oct_adj.jpg"
        print('OCT')
    elif month == 'NOV':
        earth_month = earth_nov
        earth_month_adj = earth_nov_adj
        color_oceanblack_month = color_oceanblack_nov
        earth_map = "earth_nov.jpg"
        earth_adj = "earth_nov_adj.jpg"
        print('NOV')
    elif month == 'DEC':
        earth_month = earth_dec
        earth_month_adj = earth_dec_adj
        color_oceanblack_month = color_oceanblack_dec
        earth_map = "earth_dec.jpg"
        earth_adj = "earth_dec_adj.jpg"
        print('DEC')

# Other Objects of Scene()
    if scenario==[]:
        pov_name = "tmp_" + str(int(iter)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '.ini'
    else:
        pov_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".pov"
        ini_name = "tmp_" + str(int(iter)) + "_" + str(int(scenario)) + ".ini"
        pov_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.pov'
        ini_namesingle = 'tmp_' + str(int(iter)) + '_' + str(int(scenario)) + '.ini'
    pov_file = open(pov_namesingle,
                    'w')  # scene temporary file

    pov_file.write('#version 3.7;\n'
                   '#include "colors.inc"\n'
                   '#include "functions.inc"\n\n')

    # pov_file.write('// GLOBAL SETTINGS\n'
    #         'global_settings { \n'
    #         '    assumed_gamma 1.0\n'
    #         '    radiosity { \n'
    #         '        count 1000\n'
    #         '        nearest_count 20\n'
    #         '        recursion_limit 1\n'
    #         '        normal on\n'
    #         '    }\n'
    #         '}\n')

    # ocean reflection on = 1, off = 0, original maps = 2
    if ocean_reflection == 0:
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'diffuse', 0.85, 'brilliance', 1.0)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            '                   [0.000 pigment{color rgb <0.007, 0.012, 0.03>} finish{ambient 0.0 diffuse 0.85 brilliance 1.0}]\n'
            '                   [1.000 pigment{image_map{jpeg ' +
            earth_month_adj + ' map_type 1}} finish{ambient 0.0}]\n'
            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 1: #new version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(ImagePattern('jpeg', color_oceanblack_month, "map_type", 1),
        #                           TextureMap([0.000, Pigment('color', "rgb <0.007, 0.012, 0.03>"),
        #                                       Finish('ambient', 0.0, 'reflection', "{0.05,0.25}", 'diffuse', 0.85, 'brilliance', 1.0, 'phong', 0.06, 'phong_size', 30)],
        #                                      [1.000, Pigment(ImageMap('jpeg', earth_month_adj, "map_type", 1)), Finish('ambient', 0.0)])),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           image_pattern{\n'
            '               jpeg ' + color_oceanblack_month + ' map_type 1\n'
            '           }\n'
            '           texture_map{\n'
            # '                   [0.000 pigment{color rgb <0.007, 0.03, 0.1>*0.6} finish{ambient 0.0 reflection {0.05,0.25} diffuse 0.85 brilliance 1.0 phong 0.06 phong_size 30}]\n'
            '                   [0.000 pigment{color rgb <0.0, 0.0, 0.0>} finish{ambient 0.0 reflection {0.0} diffuse 1 brilliance 0 phong 0.0}]\n'
            '                   [1.000 pigment{color rgb <1.00, 1.0, 1.0>} finish{ambient 0.0 reflection {0.0} diffuse 1 brilliance 0 phong 0.0}]\n'


            '           }\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    elif ocean_reflection == 2: #old version
        # sphere_e = Sphere([0, 0, 0], 3.5, 'scale', [1., R_Earth_pole/R_Earth, 1.],
        #                   Texture(Pigment(ImageMap('jpeg', earth_month, "map_type", 1)), Finish(
        #                       'ambient', 0.0)),
        #                   'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
        #                   'matrix', '<' +
        #                   str(rotm_earth[0][0]) + ',' + str(rotm_earth[0]
        #                                                     [1]) + ',' + str(rotm_earth[0][2]) + ','
        #                   + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
        #                                                       [1]) + ',' + str(rotm_earth[1][2]) + ','
        #                   + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')
        pov_file.write(
            '#declare sphere_e = \n'
            '   sphere {\n'
            '       <0, 0, 0>, 3.5\n'
            '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
            '       texture {\n'
            '           pigment{\n'
            '               image_map{\n'
            '                   jpeg ' + earth_month + ' map_type 1\n'
            '               }\n'
            '           }\n'
            '           finish{ambient 0.0}\n'
            '       }\n'
            '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
            '       matrix <' + str(rotm_earth[0][0]) + ',' +
            str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
            + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                                [1]) + ',' + str(rotm_earth[1][2]) + ','
            + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                                ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
            '   }\n')

    # color_oceanblack: ocean black, continents white.
    # TextureMap 0.00: set color of the ocean.
    # TextureMap 1.00: use earth map for continents

    # atmos = Difference(Sphere([0, 0, 0], 3.504, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                    Sphere([0, 0, 0], 3.501, 'scale', [
    #                           1., R_Earth_pole/R_Earth, 1.]),
    #                    Material(Texture(Pigment("rgbt 1")), Interior(Media(Scattering(5, "color White", "eccentricity 0.56"),
    #                                                                        Density('spherical', DensityMap([0.0, "rgb 0.0"], [0.5294*0.25e-6, "rgb <0.02, 0.05, 0.2>*0.07"],
    #                                                                                                        [0.5294*0.4e-6, "rgb <0.02, 0.07, 0.3>*0.32"], [
    #                                                                            0.5294*0.5e-6,   "rgb <0.08, 0.18, 0.4>*0.5"],
    #                                                                            [0.5412*0.6e-6, "rgb <0.08, 0.18, 0.4>*0.9"], [
    #                                                                            0.5471*0.65e-6,  "rgb <0.08, 0.18, 0.4>*1.5"],
    #                                                                            [0.5471*0.675e-6, "rgb <0.08, 0.18, 0.4>*4.5"], [0.5471*0.71e-6,  "rgb <0.08, 0.18, 0.4>*12"]), "scale 3.52")))),
    #                    'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z')

    # pov_file.write(
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5003\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 3.5001\n'
    #     '           scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt 1\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     '                   scattering{\n'
    #     '                       4 color White\n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     '                           [0.0 rgb 0.0]\n'
    #     '                           [0.5294*0.25e-6 rgb <0.4, 0.5, 0.6>*0.07*10]\n'
    #     '                           [0.5294*0.4e-6 rgb <0.4, 0.5, 0.6>*0.32*10]\n'
    #     '                           [0.5294*0.5e-6 rgb <0.4, 0.5, 0.6>*0.5*10]\n'
    #     '                           [0.5412*0.6e-6 rgb <0.4, 0.5, 0.6>*0.9*10]\n'
    #     '                           [0.5471*0.65e-6 rgb <0.4, 0.5, 0.6>*1.5*10]\n'
    #     '                           [0.5471*0.685e-6 rgb <0.4, 0.5, 0.6>*4.5*10]\n'
    #     '                           [0.5471*0.69e-6 rgb <0.4, 0.6, 0.8>*4*10]\n'
    #     '                       }\n'
    #     '                       scale 3.5003\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
    #     '#declare atmos = \n'
    #     '   difference {\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 1\n'
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       sphere {\n'
    #     '           <0, 0, 0>, 0.989\n' #0.989
    #     '           scale <1., 1., 1.>\n'
    #     '       }\n'
    #     '       material{\n'
    #     '           texture{\n'
    #     '               pigment{\n'
    #     '                   rgbt <1,1,1,1>\n'
    #     '               }\n'
    #     '           }\n'
    #     '           interior{\n'
    #     '               media{\n'
    #     # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
    #     '                   scattering { 4 rgb <0.4, 0.6, 0.8> \n'
    #     '                   }\n'
    #     '                   density{\n'
    #     '                       spherical\n'
    #     '                       density_map{\n'
    #     # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
    #     # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
    #     # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
    #     # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
    #     # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
    #     # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
    #     # '                           [0.0 rgb <1,1,1>*0.]\n'
    #     # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
    #     # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
    #     # '                           [0.0039 rgb <1,1,1>*0.000]\n'
    #     '                           [0.0063 rgbt 0.0]\n'
    #     '                           [0.0085 rgbt 0.1]\n'
    #     # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
    #     '                           [0.0111 rgbt 1.0]\n'
    #     '                           [0.0111 rgbt 0.0]\n'
    #     '                           [1.0 rgbt 0.0]\n'
    #     '                       }\n'
    #     '                   }\n'
    #     '               }\n'
    #     '           }\n'
    #     '       }\n'
    #     # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
    #     '       scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'
    #     # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
    #     '       hollow on\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    # pov_file.write(
    #     '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
    #     '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
    #     '#declare fBaseRadius = '+str(3.5)+';\n'
    #     '#declare fAtmoFactor = '+str(1.0114)+';\n'
    #     '#declare cAtmoColor = <1,1,1>;'
    #     '#declare M_Atmosphere_Intervals = 10;\n'
    #     '#declare M_Atmosphere_Samples = 10;\n'
    #     '#declare ATMO_STEPS = 9;\n'
    #     '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
    #     # '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare ATMO_STRENGTH = 20;\n'
    #     '#declare M_Atmosphere =\n'
    #     '   media { \n'
    #     # '	    emission 0.3\n'
    #     '       scattering { 4 color rgb <1,1,1> \n'
    #     '	    intervals M_Atmosphere_Intervals\n'
    #     '	    samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
    #     '	    method 3\n'
    #     '	    density { spherical\n'
    #     '		    color_map {\n'
    #     '		        [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
    #     '               #local I=1;#while(I<ATMO_STEPS)\n'
    #     '               #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
    #     # '               #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
    #     '               #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
    #     # '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
    #     '               [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
    #     '               #local I=I+1;#end\n'
    #     # '			    [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '               [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
    #     '			    [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
    #     '			    [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
    #     '           }\n'
    #     '       }\n'
    #     '   }\n'
    #     '#declare atmos = \n'
    #     '   sphere { < 0, 0, 0>, 1\n'
    #     '       no_shadow\n'
    #     '       hollow off\n'        
    #     '       texture { pigment { P_EmittingAtmo } } // sun-side emission\n'        
    #     '       no_shadow\n'
    #     '       hollow on\n'        
    #     '       interior { media { M_Atmosphere } }\n'        
    #     '       finish { surface_finish }\n'
    #     '       scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
    #     '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    #     '       matrix <' + str(rotm_earth[0][0]) + ',' +
    #     str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
    #                                         ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    #     '   }\n')

    #https://github.com/pyramid3d/space3d/blob/master/include/s3d_scene.inc
    pov_file.write(
        # '#declare density_input = function(x,y,z,b) {0*x+0*y+0*z+1*b}\n'
        '#declare P_EmittingAtmo =  pigment { color rgb 0 transmit 1 }'
        '#declare surface_finish = finish { ambient 0.0 diffuse 1.0 }\n'
        '#declare fBaseRadius = '+str(3.5)+';\n'
        '#declare fAtmoFactor = '+str(1.0114)+';\n'
        '#declare cAtmoColor = <1,1,1>;'
        '#declare M_Atmosphere_Intervals = 1;\n'
        '#declare M_Atmosphere_Samples = 1;\n'
        '#declare ATMO_STEPS = 9;\n'
        '#declare ATMO_EXPONENT = 8.42; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_EXPONENT = -2.0; // realistic = -2.0; cinematographic = -1.5\n'
        # '#declare ATMO_STRENGTH = 20;\n'
        '#declare ATMO_STRENGTH = 25;\n'
        # '#declare M_Atmosphere =\n'
        '#declare atmos = \n'
        # '   difference {\n'
        '   sphere {\n'
        '       <0, 0, 0>, 1\n'
        # '       scale <1., 1., 1.>\n'
        # '   }\n'
        # '       sphere {\n'
        # '           <0, 0, 0>, 0.989\n' #0.989
        # '           scale <1., 1., 1.>\n'
        # '       }\n'
        '       material{\n'
        '           texture{\n'
        '               pigment{\n'
        '                   rgbt <1,1,1,1>\n'
        '               }\n'
        '           }\n'
        '           interior{\n'
        '           media{\n'
    # '                   scattering { 4 color rgb <0.05, 0.08, 0.15> \n'
        # '               scattering { 4 rgb <0.4, 0.6, 0.8> \n'
        '               scattering { 4 rgb <0.3, 0.4, 0.5> }\n'
        '	            intervals M_Atmosphere_Intervals\n'
        '	            samples M_Atmosphere_Samples     // more: e.g.100 => smoother\n'
        '               density{\n'
        '                   spherical\n'
        '                   density_map {\n'
        '                       [0.0 color rgb <0, 0, 0> filter 0 transmit 1] // outside\n'
        '                       #local I=1;#while(I<ATMO_STEPS)\n'
        '                       #local POS = log(10*(I+1)/(ATMO_STEPS+1));\n'
        # '                           #local VAL = ATMO_STRENGTH-ATMO_STRENGTH*pow(1-POS,exp(ATMO_EXPONENT));\n'
        '                       #local VAL = pow(ATMO_STRENGTH,-(1-POS)*(fAtmoFactor-1)*'+str(R_Earth)+'/ATMO_EXPONENT);\n' #simple exponential atmospheric model: https://scipp.ucsc.edu/outreach/balloon/glost/environment3.html
        # '                           [(fAtmoFactor-1)*POS color cAtmoColor*VAL/fBaseRadius filter 1 transmit 1]\n'
        '                       [(fAtmoFactor-1)*POS color cAtmoColor*VAL filter 1 transmit 1]\n'
        '                       #local I=I+1;#end\n'
        # '			                [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*ATMO_STRENGTH filter 0 transmit 1] // surface\n'
        '                       [(fAtmoFactor-1) color cAtmoColor*0/fBaseRadius filter 0 transmit 1] // surface\n'
        '                       [1.0 color rgb 0 filter 0 transmit 1]// <0, 0, 0>] // inside\n'
        '                   }\n'
        # '                       density_map{\n'
        # # '                           [0.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0009 rgb <0.05, 0.08, 0.15>*6*0.0001]\n'
        # # '                           [0.0039 rgb <0.05, 0.08, 0.15>*6*0.001]\n'
        # # '                           [0.0063 rgb <0.05, 0.08, 0.15>*6*0.01]\n'
        # # '                           [0.0085 rgb <0.05, 0.08, 0.15>*6*0.1]\n'
        # # '                           [0.0103 rgb <0.05, 0.08, 0.15>*6*0.5]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*1.]\n'
        # # '                           [0.0111 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [1.0 rgb <0.05, 0.08, 0.15>*6*0.]\n'
        # # '                           [0.0 rgb <1,1,1>*0.]\n'
        # # '                           [0.0009 rgbt <1,1,1>*0.0001]\n'
        # # '                           [0.0039 rgbt <1,1,1>*0.001]\n'
        # # '                           [0.0039 rgb <1,1,1>*0.000]\n'
        # '                           [0.0063 rgbt 0.0]\n'
        # '                           [0.0085 rgbt 0.1]\n'
        # # '                           [0.0103 rgbt <1,1,1>*0.5]\n'
        # '                           [0.0111 rgbt 1.0]\n'
        # '                           [0.0111 rgbt 0.0]\n'
        # '                           [1.0 rgbt 0.0]\n'
        # '                       }\n'
        '                       }\n'
        '                   }\n'
        '               }\n'
        '           }\n'
        # '       scale <3.9, ' + str(3.9) + ', 3.9>\n'
        # '           scale <3.54, ' + str(R_Earth_pole/R_Earth*3.54) + ', 3.54>\n'        
        '           scale <3.5*1.0114, ' + str(R_Earth_pole/R_Earth) + '*3.5*1.0114, 3.5*1.0114>\n'
        # '       scale <3.54, ' + str(3.54) + ', 3.54>\n'
        '           hollow on\n'        
        '           no_shadow\n'
        '           finish { surface_finish }\n'
        '           rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '           matrix <' + str(rotm_earth[0][0]) + ',' +
        str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
        + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
                                            [1]) + ',' + str(rotm_earth[1][2]) + ','
        + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]
                                            ) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
        '   }\n')

        # '                           [1.0 rgb 0.]\n'
        # '                           [0.0109 rgb 0.]\n'
        # '                           [0.0109 rgb 1.]\n'
        # '                           [0.0103 rgb 0.5]\n'
        # '                           [0.0085 rgb 0.1]\n'
        # '                           [0.0063 rgb 0.01]\n'
        # '                           [0.0039 rgb 0.001]\n'
        # '                           [0.0009 rgb 0.0001]\n'
        # '                           [0.0 rgb 0.0]\n'
    
        # '                           [1.0 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.0,0.0,0.0>]\n'
        # '                           [0.0109 rgb <0.4, 0.6, 0.8>*10000]\n'
        # '                           [0.0103 rgb <0.4*0.5, 0.6*0.5, 0.8*0.5>*10000]\n'
        # '                           [0.0085 rgb <0.4*0.1, 0.6*0.1, 0.8*0.1>*10000]\n'
        # '                           [0.0063 rgb <0.4*0.01, 0.6*0.01, 0.8*0.01>*10000]\n'
        # '                           [0.0039 rgb <0.4*0.001, 0.6*0.001, 0.8*0.001>*10000]\n'
        # '                           [0.0009 rgb <0.4*0.0001, 0.6*0.0001, 0.8*0.0001>*10000]\n'
        # '                           [0.0 rgbt <0,0,0,1>*1000]\n'

    # clouds = Difference(Sphere([0, 0, 0], 3.502, 'scale', [1., R_Earth_pole/R_Earth, 1.]),
    #                     Sphere([0, 0, 0], 3.501, 'scale', [
    #                            1., R_Earth_pole/R_Earth, 1.]),
    #                     Texture(Pigment(ImagePattern('tiff', color_clouds_fileq, "map_type", 1),
    #                                     ColorMap([0.03, 'color', "rgbt <1,1,1,1>"], [1.0, 'color', "rgbt <1,1,1,0>"])),
    #                             Finish("ambient 0.0")), 'hollow on', 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                     'matrix', '<' + str(rotm_earth[0][0]) + ',' +
    #                     str(rotm_earth[0][1]) + ',' +
    #                     str(rotm_earth[0][2]) + ','
    #                     + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1]
    #                                                         [1]) + ',' + str(rotm_earth[1][2]) + ','
    #                     + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>')
    # alternate cloud map
    #     clouds =    Difference(Sphere([0,0,0],3.502, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Sphere([0,0,0],3.501, 'scale',[1.,R_Earth_pole/R_Earth,1.]),
    #                 Texture(Pigment(ImagePattern('jpeg', cloud_fraction, "map_type", 1),
    #                 ColorMap([0.66, 'color', "rgbt <1,1,1,1>"],[0.95, 'color', "rgbt <1,1,1,0>"])),
    #                 Finish("ambient 0.0")),'hollow on','rotate','-90*x','scale','<1,1,-1>','rotate','180*z',
    #                 'matrix', '<' + str(rotm_earth[0][0]) + ',' + str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    #                 + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    #                 + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ' , 0, 0, 0>')

    #0.03
    sun_dir = np.array(sun_pos_e)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    pov_file.write(
    '#declare clouds = \n'
    '   sphere {\n'
    '       <0, 0, 0>, 3.5003\n'
    '       scale <1., ' + str(R_Earth_pole/R_Earth) + ', 1.>\n'
    '       texture {\n'
    '           pigment{\n'
    '               image_pattern{\n'
    '                   jpeg ' + color_clouds_fileq + ' map_type 1\n'
    '               }\n'
    '               color_map{\n'
    '                   [0.0 color rgbt <1,1,1,1>]\n'
    '                   [0.3 color rgbt <1,1,1,1>]\n'
    '                   [0.30001 color rgbt <1,1,1,0>]\n'
    '                   [1.0 color rgbt <1,1,1,0>]\n'
    '               }\n'
    '           }\n'
    '           finish{ambient 0.0 diffuse 1}\n'
    '       }\n'
    '       hollow on\n'
    '       no_shadow\n'
    '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
    '       matrix <' + str(rotm_earth[0][0]) + ',' +
    str(rotm_earth[0][1]) + ',' + str(rotm_earth[0][2]) + ','
    + str(rotm_earth[1][0]) + ',' + str(rotm_earth[1][1]) + ',' + str(rotm_earth[1][2]) + ','
    + str(rotm_earth[2][0]) + ',' + str(rotm_earth[2][1]) + ',' + str(rotm_earth[2][2]) + ', 0, 0, 0>\n'
    '       clipped_by { plane { <' + str(-sun_dir[0]) + ',' + str(-sun_dir[1]) + ',' + str(-sun_dir[2]) + '>, 0 } }\n'
    '   }\n')
    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # ------------------------------------------------------------------------------------------------------
    # Moon Sphere
    # scaling factor for the Earth Moon environment (Earth primary central body)
    scale_m = R_Moon/R_Earth*3.5
    # imported values range from 0 to 1, transform to -1 to 1, then from -32768 to 32767, in this range, multiply by factor 0.5 to convert into meters
    # ele = 'ele=function{pigment{image_map{tiff ".Maps/moon/elevation_20.tiff" map_type 1 interpolate 2}}}'

 # Accuracy Parameter
    sc_dist = np.array(sc_pos) - np.array(moon_pos_e)
    dist = norm(sc_dist)    # in km
    if dist <= 3000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_1.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 10000:
        acc = 0.001       
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_2.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj_10k.tiff") + '"'
    elif dist <= 20000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 30000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 70000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    elif dist <= 100000:
        acc = 0.00001        
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
    else:
        acc = 0.001     
        elevation_fileq = '"' + \
            os.path.join(os.getcwd(), "Maps/moon/elevation_20.tiff") + '"'        
        moon_adj = '"' + os.path.join(os.getcwd(), "Maps/moon/moon_adj.tiff") + '"'
        
    pov_file.write(
    '#declare ele = \n'
    '   function{\n'
    '       pigment{\n'
    '           image_map{\n'
    '               tiff ' + elevation_fileq + ' map_type 1 interpolate 2\n'
    '           }\n'
    '       }\n'
    '   }\n')
    

#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture(Pigment(ImageMap('tiff', moon_darklight, "map_type", 1)),
#                 Finish('ambient', "0", 'diffuse', 0.7, 'brilliance', 1.5,)), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
        # The rotation of the sphere is done by using the matrix parameter (https://www.povray.org/documentation/view/3.6.0/49/) which takes directly the elements of the rotation matrix to compute the rotation. The rotation matrix is obtained through spice and it is from inertial (J2000) to body fixed (IAU_MOON)
        # adjust properties so we can use a single light source
#     sphere_m =  Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
#                 ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)), 'accuracy', str(acc), 'max_gradient', 1.2, # str(acc)
#                 Texture( ImagePattern( 'tiff', moon_craters2, "map_type", 1),
#                     TextureMap([0.000,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.8,'brilliance',1.2)],
#                                [0.001,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                [0.999,Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                     Finish('ambient', 0.0,'diffuse',0.35)],
#                                 [1.000, Pigment(ImageMap('tiff',moon_adj,"map_type",1)),
#                                  Finish('ambient',0.0,'diffuse',0.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate','180*z',
#                 'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
#                 + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1][1]) + ',' + str(rotm_moon[1][2]) + ','
#                 + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
#                 'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')
    # sphere_m = Isosurface("function {f_sphere(x,y,z," + str(scale_m) + ") -  (ele(x,y,z).gray-0.5)*(2*32.767/2/" + str(R_Earth/3.5) + ")}",
    #                       ContainedBy(Sphere(0, (1 + highest_pt / R_Moon)*scale_m)
    #                                   ), 'accuracy', str(acc), 'max_gradient', 1.2,  # str(acc)
    #                       Texture(ImagePattern('tiff', moon_adj, "map_type", 1),
    #                               TextureMap([0.0, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.2, 'brilliance', 1.2)],
    #                                          [0.2, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.25)],
    #                                          [0.5, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.4)],
    #                                          [1.000, Pigment(ImageMap('tiff', moon_adj, "map_type", 1)),
    #                                           Finish('ambient', 0.0, 'diffuse', 0.9, 'brilliance', 1.2)])), 'rotate', '-90*x', 'scale', '<1,1,-1>', 'rotate', '180*z',
    #                       'matrix', '<' + str(rotm_moon[0][0]) + ',' + str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
    #                        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
    #                                                          [1]) + ',' + str(rotm_moon[1][2]) + ','
    #                       + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
    #                                                          ) + ',' + str(rotm_moon[2][2]) + ' , 0, 0, 0>',
    #                       'translate', '<'+str(moon_pos_e[0]/R_Earth*3.5)+','+str(moon_pos_e[1]/R_Earth*3.5)+','+str(moon_pos_e[2]/R_Earth*3.5)+'>')

    pov_file.write(
        '#declare sphere_m = \n'
        '   isosurface {\n'
        '       function {\n'
        '           f_sphere(x,y,z,' + str(scale_m) +
        ') -  (ele(x,y,z).gray-0.5)*(2*32.767/2/' + str(R_Earth/3.5) + ')\n'
        '           }\n'
        '       contained_by {\n'
        '           sphere {0, (1 + ' + str(highest_pt /
                                            R_Moon) + ')*' + str(scale_m) + '}\n'
        '           }\n'
        '       accuracy ' + str(acc) + ' max_gradient 2.0\n'
        '       texture {\n'
        '           image_pattern{\n'
        '               tiff ' + moon_adj + ' map_type 1 interpolate 2\n'
        '           }\n'
        '           texture_map{\n'
        '               [0.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.2 brilliance 1.2}]\n'
        '               [0.2 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.25}]\n'
        '               [0.5 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.4}]\n'
        '               [1.0 pigment{image_map{tiff ' + moon_adj +
        ' map_type 1}} finish{ambient 0.0 diffuse 0.9 brilliance 1.2}]\n'
        '           }\n'
        '       }\n'
        '       rotate -90*x scale <1,1,-1> rotate 180*z\n'
        '       matrix <' + str(rotm_moon[0][0]) + ',' +
        str(rotm_moon[0][1]) + ',' + str(rotm_moon[0][2]) + ','
        + str(rotm_moon[1][0]) + ',' + str(rotm_moon[1]
                                           [1]) + ',' + str(rotm_moon[1][2]) + ','
        + str(rotm_moon[2][0]) + ',' + str(rotm_moon[2][1]
                                           ) + ',' + str(rotm_moon[2][2]) + ', 0, 0, 0>\n'
        '       translate <' + str(moon_pos_e[0]/R_Earth*3.5) + ',' + str(
            moon_pos_e[1]/R_Earth*3.5) + ',' + str(moon_pos_e[2]/R_Earth*3.5) + '>\n'
        '   }\n')

    # adjust regions -> 0,0.4,0.8,1.0
    # no need for moon_Craters2, just use moon_adj

    # Star Sky Sphere - Not used because the Moon and the Earth are far brighter than stars, making the star map completely black (unless it is a parallax observation)
    # stars =     Sphere([0,0,0], 1, 'hollow on', Texture( Pigment( ImageMap( 'tiff', color_star_fileq, "map_type", 1 ))),
    #             'rotate', '-90*x', 'scale', '<1, 1, -1>', 'rotate', '180*z', 'scale', 1E+7,
    #             'translate', '<'+str(star_pos_e[0]/R_Earth*3.5)+','+str(star_pos_e[1]/R_Earth*3.5)+','+str(star_pos_e[2]/R_Earth*3.5)+'>')

    # scene = Scene(camera, objects=[light, sphere_e, clouds, sphere_m, atmos], included=[
    #     "functions.inc", "colors.inc"], declares=[ele])

    look_at_object, focal_len, cam_width, cam_height, px = camera_definition
    fov_angle = 2*math.atan2(px*cam_width/2, focal_len)*180/math.pi

    if look_at_object == 'EARTH':
        # pov_file.write('object {sphere_e}\n')
        # pov_file.write('object {atmos}\n')
        pov_file.write('object {clouds}\n')
    elif look_at_object == 'MOON':
        pov_file.write('object {sphere_m}\n')

    if look_at_object == 'EARTH':
        print('EARTH IMAGE')
        if scenario==[]:
            image_name = "earth_img_CLOUDMASK" + str(int(iter)) + ".png"
        else:
            image_name = "earth_img_CLOUDMASK" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            look_at = [0, 0, 0]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]
        
    elif look_at_object == 'MOON':
        print('MOON IMAGE')
        if scenario==[]:
            image_name = "moon_img_" + str(int(iter)) + ".png"
        else:
            image_name = "moon_img_" + str(int(iter)) + "_" + str(int(scenario)) + ".png"
        if len(lookat_input)==0:
            # if dist-R_Moon<= 5000:
            #     sc_pos_rel_moon =  sc_pos-moon_pos_e
            #     norm_pos_rel_moon = np.linalg.norm(sc_pos_rel_moon)
            #     look_at = [moon_pos_e[0]/R_Earth*3.5 + sc_pos_rel_moon[0]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[1] /
            #             R_Earth*3.5 + sc_pos_rel_moon[1]/norm_pos_rel_moon*R_Moon/R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5 + sc_pos_rel_moon[2]/norm_pos_rel_moon*R_Moon/R_Earth*3.5]
            # else:
            look_at = [moon_pos_e[0]/R_Earth*3.5, moon_pos_e[1] /
                    R_Earth*3.5, moon_pos_e[2]/R_Earth*3.5]
        else:
            lookat_input = lookat_input.flatten()
            look_at = [lookat_input[0]/R_Earth*3.5, lookat_input[1] /
                    R_Earth*3.5, lookat_input[2]/R_Earth*3.5]

    scene_file = os.path.join(os.getcwd(), image_name)
    scene_fileq = '"' + scene_file + '"'

    # camera = Camera('right', '<-' + str(cam_width/cam_height) + ',0,0>', 'sky', 'z', 'direction', 'z', 'angle', str(fov_angle),
    #                 'location', [sc_pos[0]/R_Earth*3.5, sc_pos[1] /
    #                              R_Earth*3.5, sc_pos[2]/R_Earth*3.5], 'up', 'z',
    #                 'look_at', look_at)

    # Camera with FOV and Resolution Implementation
    if len(sky_vec_input)==0:    
        sc2obj = np.array(look_at)-np.array(sc_pos)/R_Earth*3.5
        sc2obj = sc2obj/np.linalg.norm(sc2obj)
        # print(sc2obj)
        if norm(np.cross(sc2obj, np.array((0,0,1))))<1e-14:            
            x_b = np.array((1,0,0))           
            z_b = np.cross(x_b, sc2obj)
            sky_vec = '<' + str(z_b[0]) + ',' + str(z_b[1]) + ',' + str(z_b[2]) + '>'
            # print(sky_vec)
            # error('s')
        else:
            sky_vec = 'z'
        # Axis z the vertical of the camera plane
    else:
        sky_vec = '<' + str(sky_vec_input[0]) + ',' + str(sky_vec_input[1]) + ',' + str(sky_vec_input[2]) + '>'

    pov_file.write('camera {\n'
                   '    right <-' + str(cam_width/cam_height) + ',0,0>\n'
                   '    up z\n'
                   '    direction z\n'
                   '    sky ' + sky_vec + '\n'
                   '    angle ' + str(fov_angle) + '\n'
                   '    location <' + str(sc_pos[0]/R_Earth*3.5) + ',' + str(
                       sc_pos[1]/R_Earth*3.5) + ',' + str(sc_pos[2]/R_Earth*3.5) + '>\n'
                   '    look_at <' +
                   str(look_at[0]) + ',' + str(look_at[1]) +
                   ',' + str(look_at[2]) + '>\n'
                   '}\n')

    # light source
    # light = LightSource([sun_pos_e[0]/R_Earth*3.5, sun_pos_e[1] /
    #                     R_Earth*3.5, sun_pos_e[2]/R_Earth*3.5], 'color', "rgb 2.4")

    pov_file.write('light_source { \n'
                   '    <' + str(sun_pos_e[0]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[1]/R_Earth*3.5) + ',' +
                   str(sun_pos_e[2]/R_Earth*3.5) + '>\n'
                   '    color rgb 2.4\n'
                   '}\n')

    pov_file.close()

    # Begin render--------------------------
    # width and height define the resolution of the image generated
    # scene.render(scene_file, width=cam_width, height=cam_height)

    ini_file = open(ini_namesingle,
                    'w')  # .ini temporary file

    ini_file.write(
        'Width=' + str(cam_width) + '\n'
        'Height=' + str(cam_height) + '\n'
        # 'Max_Image_Buffer_Memory=' + str(100000) + '\n'
        'Antialias=On\n'
        'Antialias_Threshold=0.0\n'
        'Antialias_Depth=3\n'
        'Input_File_Name=' + '"' +
        os.path.join(os.getcwd(), pov_name) + '"' + '\n'
        'Output_File_Name=' + scene_fileq)

    ini_file.close()

    # Call POV-Ray
    if platform == 'win32':  # if on Windows platform
        os.system('C:/PROGRA~1/POV-Ray/v3.7/bin/pvengine.exe /NR /RENDER ' +
                  '"' + os.path.join(os.getcwd(), ini_name) + '" /EXIT')
    else:
        povray_location = subprocess.getoutput('which povray')
        print(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')
        os.system(povray_location + ' "' +
                  os.path.join(os.getcwd(), ini_name) + '"')

    stop = timeit.default_timer()
    print('End - Total Time: ' + str("{:.2f}".format(stop - start)) + ' s')

# OpenCV Plot ----------------------------------------------------------------------------------------------------------------------------------------
    # plt.figure(figsize=(9.5, 5.5))
    # plt.tight_layout()
    # color_src = cv2.imread(scene_file)
    # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
    # color_plot = plt.imshow(color_src)
    # plt.axis('off')
    # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
    # plt.show()
# end render------------------------------------------------
    return scene_file, earth_map, cloud_map