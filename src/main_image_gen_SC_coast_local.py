import os
import sys
from functions import *
import spiceypy as spice  # Spice library for computation of orbital elements
import numpy as np
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import timing
from skimage.util import random_noise
from PIL import Image
kernels_path = os.path.join(os.getcwd(), "Kernels")
spice.furnsh(os.path.join(kernels_path, "naif0012.tls"))  # Time kernel
spice.furnsh(os.path.join(kernels_path, "de441.bsp"))  # Ephemeris kernel
spice.furnsh(os.path.join(kernels_path, "pck00010.tpc"))  # Frames kernel
# Frames kernel
spice.furnsh(os.path.join(kernels_path, "moon_pa_de403_1950-2198.bpc"))
spice.furnsh(os.path.join(kernels_path, "moon_060721.tf"))  # Frames kernel

# for training_idx in range(1):
# training_idx = int(sys.argv[1])  # get task ID
# small_idx = int(sys.argv[2])  # get smallest task ID
# large_idx = int(sys.argv[3])  # get largest task ID
# total_idx = int(sys.argv[4])  # get total number of tasks
training_idx = 0  # get task ID 0
small_idx = 0  # get smallest task ID 0
large_idx = 0  # get largest task ID 1
total_idx = 1 # get total number of tasks 2
# CHOOSE between these various cases
# EARTH
# case 1: Apollo 17's Blue Marble, 12/7/1972 10AM
# case 2: 1990 Galileo family portrait 12/11/1990 07:50
# case 3: Apollo 15 7/26/1971
# MOON
# case 101: orion flyby 11/21/22 10:00
# case 102: NASA's Cassini 08/17/1999
# case 103: orion 11/22/2022
# num_theta = 36
# num_phi = 17

# theta_all = np.linspace(0, 350, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-80, 80, num=num_phi)*(math.pi/180)
# dist_all = np.hstack((25000, 50000, 75000, 100000, 150000,
#                      200000, 250000, 300000, 350000))

# Train
# num_theta = 36
# num_phi = 18
# theta_all = np.linspace(0, 350, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-85, 85, num=num_phi)*(math.pi/180)
# dist_all = np.hstack((40000,60000,80000,100000,150000,200000,250000,300000))

# Check performance
# num_theta = 36
# num_phi = 17
# theta_all = np.linspace(5, 355, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-80, 80, num=num_phi)*(math.pi/180)
# dist_all = np.hstack((50000,70000,90000,125000,175000,225000,275000))

num_theta = 1
num_phi = 2

theta_all = np.linspace(0, 330, num_theta) * (math.pi/180)
phi_all   = np.linspace(-60, 60, num_phi) * (math.pi/180)

dist_all = np.array([75000])

#List of cloud maps
# list_hours = np.arange(0,24,3)
# list_days = np.arange(1,8,1)
# list_months = np.arange(1,13,1)
# list_hours = np.arange(0,24,3)
# list_days = np.arange(1,2,1)
# list_months = np.array((2,5))

# Train
# list_hours = np.arange(0,24,3)
# list_days = np.arange(1,8,1)
# list_months = np.array((2,5,8,11))

# Test
# list_hours = np.arange(0,24,3)
# list_days = np.arange(1,8,1)
# list_months = np.array((1,3,4,6,7,9,10,12))

list_hours  = np.array([0, 12])
list_days   = np.array([5])
list_months = np.array([1, 7])

table_day_hour = np.zeros((list_days.size*list_hours.size,2))
idx_row = -1
for idx_day in range(list_days.size):
    for idx_hour in range(list_hours.size):
        idx_row = idx_row+1
        table_day_hour[idx_row,:] = [list_days[idx_day],list_hours[idx_hour]]

# #In range
# num_theta = 36
# num_phi = 18
# theta_all = np.linspace(2.5, 352.5, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-87.5, 82.5, num=num_phi)*(math.pi/180)
# dist_all = np.hstack((4000,6250,8500,12500,17500,
#                       22500,27500,32500,37500,42500,47500,52500,57500,62500,67500))
# #Above
# dist_all = np.hstack((75000,80000,85000,90000,95000,100000))
#Below
# dist_all = np.hstack((2500,2600,2700,2800,2900,3000,3100))

# all_thetaphidist = np.zeros((len(theta_all)*len(phi_all) *
#                              len(dist_all), 3))

# num_theta = 36
# num_phi = 17

# theta_all = np.linspace(0, 350, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-80, 80, num=num_phi)*(math.pi/180)
# num_theta = 72
# num_phi = 35
# theta_all = np.linspace(0, 355, num=num_theta)*(math.pi/180)
# phi_all = np.linspace(-85, 85, num=num_phi)*(math.pi/180)
# dist_all = np.hstack((25000, 50000, 75000, 100000, 150000,
#                      200000, 250000, 300000, 350000))

# 100*(math.pi/180) #theta 0<x<360
all_thetaphidist = np.zeros((theta_all.size*phi_all.size *
                             dist_all.size+2*dist_all.size, 3))

idx_image = 0
for dist in dist_all:
    all_thetaphidist[idx_image, :] = [0, -math.pi/2, dist]
    idx_image += 1
    for theta in theta_all:
        for phi in phi_all:
            all_thetaphidist[idx_image, :] = [theta, phi, dist]
            idx_image += 1
    all_thetaphidist[idx_image, :] = [0, math.pi/2, dist]
    idx_image += 1

total_cases = np.size(all_thetaphidist, 0)

#Train
# num_repeat_clouds = 7
#Test
num_repeat_clouds = 1

idx_random_date = np.random.randint(low=0, high=np.size(table_day_hour,axis=0), size=(total_cases*num_repeat_clouds), dtype=int)

vec_cases = np.arange(training_idx, total_cases, total_idx)
# if training_idx < large_idx:
#     vec_cases = range(int(np.floor(total_cases/total_idx)*(training_idx-small_idx)),
#                       int(np.floor(total_cases/total_idx)*(training_idx+1-small_idx)))
# else:
#     vec_cases = range(int(np.floor(total_cases/total_idx) *
#                       (training_idx-small_idx)), total_cases)

# missing_indices = [1004, 10162, 10179, 10191, 10244, 10371, 10385, 10551, 10633, 10656,
# 10731, 10742, 10829, 1101, 11062, 1124, 11439, 1165, 11662, 11834,
# 976, 983, 9886, 9923]
for idx_month,month in enumerate(list_months):    
    for idx_repeat in range(num_repeat_clouds):
        for idx_case in vec_cases:
            idx_count = (total_cases*num_repeat_clouds)*idx_month+total_cases*idx_repeat+idx_case
            idx_dayhour = total_cases*idx_repeat+idx_case
            day = table_day_hour[idx_random_date[idx_dayhour],0]
            print(day)
            print(table_day_hour)
            hour = table_day_hour[idx_random_date[idx_dayhour],1]
            [sim_date, camera_definition, sc_pos, iter,
                reflection, mission_specifics] = case_type(3000, training_idx=idx_count, state=all_thetaphidist[idx_case, :], month=month, day=day, hour=hour)

            sun_sp = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
            sun_pos = [sun_sp[0][0], sun_sp[0][1], sun_sp[0][2]]

            moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
            moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]

            central_pos = np.array((0,0,0))

            central2sun = np.vstack(sun_pos-central_pos)
            central2sc = np.vstack(sc_pos-central_pos)
            solar_angle = np.arccos(np.sum(
            central2sun*central2sc)/np.linalg.norm(central2sun)/np.linalg.norm(central2sc))*180/np.pi

            if solar_angle<135:
            # generate image of the earth/moon
                scene_file, earth_map, cloud_map = gen_moon_earth(
                    sc_pos, sim_date, camera_definition, iter, reflection, month_number=month, day=day, hour=hour)
            # generate mask earth image B/W
                scene_file, earth_map, cloud_map = gen_moon_earthMASK(
                    sc_pos, sim_date, camera_definition, iter, reflection, month_number=month, day=day, hour=hour)
                scene_file, earth_map, cloud_map = gen_moon_earthCLOUDMASK(
                    sc_pos, sim_date, camera_definition, iter, reflection, month_number=month, day=day, hour=hour)
                print(scene_file)
                print(earth_map)
                print(cloud_map)
                
            # plt.figure(figsize=(9.5, 5.5))
            # plt.tight_layout()
            # color_src = cv2.imread(scene_file)
            # color_src = cv2.cvtColor(color_src, cv2.COLOR_BGR2RGB)  # cv2 work in BGR and matplotlib work in RGB, so we need to convert order of colors
            # color_plot = plt.imshow(color_src)
            # plt.axis('off')
            # plt.savefig(scene_file,bbox_inches='tight', dpi=600, transparent=True, pad_inches=0)
            # plt.show()
            # error()

                print('Saving data image {}'.format(idx_case))
                np.savez("./image_gen_data_{}.npz".format(int(idx_count)), scene_file=scene_file, moon_pos=moon_pos, sun_pos=sun_pos, sim_date=sim_date,
                        camera_definition=camera_definition, iter=iter, reflection=reflection, mission_specifics=mission_specifics,
                        month=month, day=day, hour=hour, num_repeat_clouds=num_repeat_clouds, table_day_hour=table_day_hour, idx_random_date=idx_random_date,
                        idx_case=idx_case, list_months=list_months, list_days=list_days, list_hours=list_hours, earth_map=earth_map, cloud_map=cloud_map)

print('Saving data all cases')
np.savez("./image_gen_allcases.npz", all_thetaphidist=all_thetaphidist, sun_pos=sun_pos,
         moon_pos=moon_pos, sim_date=sim_date, camera_definition=camera_definition, reflection=reflection,
         num_repeat_clouds=num_repeat_clouds, table_day_hour=table_day_hour, idx_random_date=idx_random_date,
         list_months=list_months, list_days=list_days, list_hours=list_hours)

# scene_file = 'moon_img_0.png'

# fig3d_idx = training_idx+2
# num_craters = 10000

# # image overlay
# plt.figure(fig3d_idx, figsize=(15, 15))
# img = cv2.imread(scene_file)
# # if image has different number of pixels than cam_width/height
# [img_width, img_height] = get_num_pixels(scene_file)
# plt.imshow(img, cmap=plt.cm.gray)

# start = timeit.default_timer()
# # import excel file
# columns = ["CRATER_ID", "LAT_ELLI_IMG", "LON_ELLI_IMG", "DIAM_CIRC_IMG",
#            "DIAM_ELLI_MAJOR_IMG", "DIAM_ELLI_MINOR_IMG", "DIAM_ELLI_ANGLE_IMG"]
# file = pd.read_csv(
#     "./Maps/moon/lunar_crater_database_robbins_2018.csv", usecols=columns)
# # import name, diameter, latitude, and longitude of each crater
# # crater = file.CRATER_ID
# lat = np.array(file.LAT_ELLI_IMG)
# long = np.array(file.LON_ELLI_IMG)
# long[np.where(long > 180.)] += -360.  # from -180 to 180
# semimajor = file.DIAM_ELLI_MAJOR_IMG/2
# semiminor = file.DIAM_ELLI_MINOR_IMG/2
# diameter = file.DIAM_CIRC_IMG
# angle = file.DIAM_ELLI_ANGLE_IMG
# # sort
# idx_diam = np.array(diameter).argsort()[::-1]
# diam_sorted = diameter[idx_diam]
# lat_sorted = lat[idx_diam]
# long_sorted = long[idx_diam]
# major_sorted = np.array(semimajor[idx_diam])
# minor_sorted = np.array(semiminor[idx_diam])
# angle_sorted = np.array(angle[idx_diam])
# # fill empty semi-axes and angles
# major_sorted[np.isnan(major_sorted)
#              ] = diam_sorted[np.isnan(major_sorted)]/2
# minor_sorted[np.isnan(minor_sorted)
#              ] = diam_sorted[np.isnan(minor_sorted)]/2
# angle_sorted[np.isnan(angle_sorted)] = 0.
# stop = timeit.default_timer()
# # alternate image
# elevation_file = './Maps/moon/elevation_20.tiff'
# width, height = get_num_pixels(elevation_file)
# # # #conversion from degrees to pixels
# lat_plot = height/180 * (90 - lat_sorted)
# long_plot = width/2 * long_sorted/180 + width/2

# plt.figure(1, figsize=(9.5, 5.5))
# img_el = cv2.imread(elevation_file)
# plt.imshow(img_el, cmap=plt.cm.gray)
# # semi-axes of moon
# # a = 1739.088
# # b = 1737.37
# # c = 1734.969
# a = 1737.1513  # km, from "A New Global Database of Lunar Impact Craters"
# b = 1737.1513  # km
# c = 1735.6576  # km
# # plotting ellipse
# theta = np.linspace(0, 2*np.pi, 30)
# max_craters = np.min([num_craters, len(diam_sorted)])

# # Preallocate crater info
# # pixel coordinates from top left corner
# store_centers = np.zeros((max_craters, 2))  # center coordinates
# # 0, 1, whether center is visible or not
# store_center_visible = np.hstack((np.zeros(max_craters)))
# # rim bounds: left, right, top, bottom bounds around crater
# store_rim_bounds = np.zeros((max_craters, 4))  # bounding box
# # 0, 1 (partially) or 2 (fully), whether crater is visible
# store_visible = np.vstack((np.zeros(max_craters)))
# # x and z pixel coordinates of crater rim
# store_rims = [np.zeros((2, 1)) for v in range(max_craters)]
# # index of crater
# store_crater_sorted_idx = np.hstack((np.zeros(max_craters)))
# # crater diameter
# store_diam_km = np.vstack((np.zeros(max_craters)))
# # crater angle with Sun, deg
# store_angle_sun_deg = np.vstack((np.zeros(max_craters)))

# for j in range(0, max_craters)[::-1]:
#     if np.remainder(j, 100) == 0:
#         print('Crater {}'.format(j))
#     coordinates_center, coordinates_rim, lat_points, lon_points = ellipse_on_ellipsoid(
#         lat_sorted[j], long_sorted[j], major_sorted[j], minor_sorted[j], angle_sorted[j])

#     # plot crater center
#     plt.figure(1)
#     plt.plot(long_plot[j], lat_plot[j],
#              color='r', marker='o', linestyle='None', markersize=1.0)  # for 2D

#     # conversion from degrees to pixels
#     lat_pointsdeg = lat_points*180/np.pi
#     lon_pointsdeg = lon_points*180/np.pi
#     lon_pointsdeg[np.where(lon_pointsdeg > 180.)] += -360.
#     latpoints_plot = height/180 * (90 - lat_pointsdeg)
#     lonpoints_plot = width/2 * lon_pointsdeg/180 + width/2

#     plt.figure(1)
#     plt.plot(lonpoints_plot, latpoints_plot,
#              color='r', marker='o', linestyle='None', markersize=1.0)  # for 2D

#     center_x, center_z, select_x, select_z, center_visible, visible, angle_sun_deg = crater_mapping(
#         sim_date, mission_specifics, camera_definition, coordinates_rim, coordinates_center, img_width, img_height, fig3d_idx)

#     if visible > 0:  # if fully or partially visible, store pixel values
#         store_centers[j, :] = [center_x, center_z]
#         store_center_visible[j] = center_visible
#         store_visible[j] = visible  # 0, 1 (partially) or 2 (fully)
#         if center_visible == 1:  # concatenate center coordinates
#             all_x = np.hstack((select_x, center_x))
#             all_z = np.hstack((select_z, center_z))
#         else:
#             all_x = select_x
#             all_z = select_z
#         store_rim_bounds[j, :] = [np.min(all_x), np.max(
#             all_x), np.min(all_z), np.max(all_z)]
#         # two rows, as many columns as visible points along the rim
#         store_rims[j] = np.vstack((select_x, select_z))
#         store_crater_sorted_idx[j] = j
#         store_diam_km[j] = diam_sorted[j]
#         store_angle_sun_deg[j] = angle_sun_deg

#         # plot crater bounds
#         fig3D = plt.figure(fig3d_idx)
#         plt.plot(store_rim_bounds[j, 0]*np.hstack(np.ones(2)), np.hstack((store_rim_bounds[j, 2], store_rim_bounds[j, 3])),
#                  color='blue', linewidth=0.5)  # for 2D
#         plt.plot(store_rim_bounds[j, 1]*np.hstack(np.ones(2)), np.hstack((store_rim_bounds[j, 2], store_rim_bounds[j, 3])),
#                  color='blue', linewidth=0.5)  # for 2D
#         plt.plot(np.hstack((store_rim_bounds[j, 0], store_rim_bounds[j, 1])), store_rim_bounds[j, 2]*np.hstack(np.ones(2)),
#                  color='blue', linewidth=0.5)  # for 2D
#         plt.plot(np.hstack((store_rim_bounds[j, 0], store_rim_bounds[j, 1])), store_rim_bounds[j, 3]*np.hstack(np.ones(2)),
#                  color='blue', linewidth=0.5)  # for 2D

#     else:  # if not visible, remove from stored variables
#         store_centers = np.delete(store_centers, j, axis=0)
#         store_center_visible = np.delete(store_center_visible, j)
#         store_visible = np.delete(store_visible, j)
#         store_rim_bounds = np.delete(store_rim_bounds, j, axis=0)
#         store_rims.pop(j)
#         store_crater_sorted_idx = np.delete(store_crater_sorted_idx, j)
#         store_diam_km = np.delete(store_diam_km, j)
#         store_angle_sun_deg = np.delete(store_angle_sun_deg, j)

# print('Saving training image')
# fig3D.savefig('./Training/img_3D_{}.png'.format(int(training_idx)),
#               dpi=500, bbox_inches='tight')
print('End of script')
# ###
