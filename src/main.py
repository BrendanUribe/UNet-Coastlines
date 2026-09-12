import os
from src.functions import *
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

# # camera definition
# cam_width = 2048  # number pixels in horizontal axis
# cam_height = 1536  # number pixels in vertical axis
# focal_len = 35  # mm, focal length
# px = 4.96E-3  # mm, pixel size
# camera_definition = [focal_len, cam_width, cam_height, px]
# map_x = 100  # km, map length in x-axis
# map_y = 200  # km, map length in y-axis
# pos_cam = [20, 10, 150] # camera location, km
# lookat_y = 30 # look_at y-coordinate, km
# lookat_x_upper, lookat_x_lower = find_x_limits(
#     pos_cam, lookat_y, map_x, map_y, focal_len, cam_width, cam_height, px) # function to find upper and lower limits of look_at x-coordinate
# pos_lookat = [lookat_x_upper, lookat_y, 0]
# gen_plane(pos_cam, pos_lookat, camera_definition,
#           0, map_x, map_y) # generate image looking at upper limit
# pos_lookat = [lookat_x_upper+1, lookat_y, 0]
# gen_plane(pos_cam, pos_lookat, camera_definition,
#           1, map_x, map_y) # generate image looking at upper limit plus some quantity
# pos_lookat = [lookat_x_lower, lookat_y, 0]
# gen_plane(pos_cam, pos_lookat, camera_definition,
#           2, map_x, map_y) # generate image looking at lower limit
# pos_lookat = [lookat_x_lower-1, lookat_y, 0]
# gen_plane(pos_cam, pos_lookat, camera_definition,
#           3, map_x, map_y) # generate image looking at lower limit minus some quantity
# pos_lookat = [lookat_x_upper+1, lookat_y, 0]
# gen_plane([0, 0, 1000], [0, 0, 0], camera_definition,
#           4, map_x, map_y) # generate image of dummy plane seen from above
# error('s')

# for training_idx in range(1):
#     # CHOOSE between these various cases
#     # EARTH
#     # case 1: Apollo 17's Blue Marble, 12/7/1972 10AM
#     # case 2: 1990 Galileo family portrait 12/11/1990 07:50
#     # case 3: Apollo 15 7/26/1971
#     # MOON
#     # case 101: orion flyby 11/21/22 10:00
#     # case 102: NASA's Cassini 08/17/1999
#     # case 103: orion 11/22/2022
#     [sim_date, camera_definition, sc_pos, iter,
#         reflection, mission_specifics] = case_type(1000, training_idx)

#     # generate image of the earth/moon
#     scene_file = gen_moon_earth(
#         sc_pos, sim_date, camera_definition, iter, reflection)
#     # scene_file = 'moon_img_0.png'

#     fig3d_idx = training_idx+2
#     num_craters = 10000

#     # image overlay
#     plt.figure(fig3d_idx, figsize=(15, 15))
#     img = cv2.imread(scene_file)
#     # if image has different number of pixels than cam_width/height
#     [img_width, img_height] = get_num_pixels(scene_file)
#     plt.imshow(img, cmap=plt.cm.gray)

#     start = timeit.default_timer()
#     # import excel file
#     columns = ["CRATER_ID", "LAT_ELLI_IMG", "LON_ELLI_IMG", "DIAM_CIRC_IMG",
#                "DIAM_ELLI_MAJOR_IMG", "DIAM_ELLI_MINOR_IMG", "DIAM_ELLI_ANGLE_IMG"]
#     file = pd.read_csv(
#         "./Maps/moon/lunar_crater_database_robbins_2018.csv", usecols=columns)
#     # import name, diameter, latitude, and longitude of each crater
#     # crater = file.CRATER_ID
#     lat = np.array(file.LAT_ELLI_IMG)
#     long = np.array(file.LON_ELLI_IMG)
#     long[np.where(long > 180.)] += -360.  # from -180 to 180
#     semimajor = file.DIAM_ELLI_MAJOR_IMG/2
#     semiminor = file.DIAM_ELLI_MINOR_IMG/2
#     diameter = file.DIAM_CIRC_IMG
#     angle = file.DIAM_ELLI_ANGLE_IMG
#     # sort
#     idx_diam = np.array(diameter).argsort()[::-1]
#     diam_sorted = diameter[idx_diam]
#     lat_sorted = lat[idx_diam]
#     long_sorted = long[idx_diam]
#     major_sorted = np.array(semimajor[idx_diam])
#     minor_sorted = np.array(semiminor[idx_diam])
#     angle_sorted = np.array(angle[idx_diam])
#     # fill empty semi-axes and angles
#     major_sorted[np.isnan(major_sorted)
#                  ] = diam_sorted[np.isnan(major_sorted)]/2
#     minor_sorted[np.isnan(minor_sorted)
#                  ] = diam_sorted[np.isnan(minor_sorted)]/2
#     angle_sorted[np.isnan(angle_sorted)] = 0.
#     stop = timeit.default_timer()
#     # alternate image
#     elevation_file = './Maps/moon/elevation_20.tiff'
#     [width, height] = get_num_pixels(elevation_file)
#     # # #conversion from degrees to pixels
#     lat_plot = height/180 * (90 - lat_sorted)
#     long_plot = width/2 * long_sorted/180 + width/2

#     plt.figure(1, figsize=(9.5, 5.5))
#     img_el = cv2.imread(elevation_file)
#     plt.imshow(img_el, cmap=plt.cm.gray)
#     # semi-axes of moon
#     # a = 1739.088
#     # b = 1737.37
#     # c = 1734.969
#     a = 1737.1513  # km, from "A New Global Database of Lunar Impact Craters"
#     b = 1737.1513  # km
#     c = 1735.6576  # km
#     # plotting ellipse
#     theta = np.linspace(0, 2*np.pi, 30)
#     max_craters = np.min([num_craters, len(diam_sorted)])

#     # Preallocate crater info
#     # pixel coordinates from top left corner
#     store_centers = np.zeros((max_craters, 2))  # center coordinates
#     # 0, 1, whether center is visible or not
#     store_center_visible = np.hstack((np.zeros(max_craters)))
#     # rim bounds: left, right, top, bottom bounds around crater
#     store_rim_bounds = np.zeros((max_craters, 4))  # bounding box
#     # 0, 1 (partially) or 2 (fully), whether crater is visible
#     store_visible = np.vstack((np.zeros(max_craters)))
#     # x and z pixel coordinates of crater rim
#     store_rims = [np.zeros((2, 1)) for v in range(max_craters)]
#     # index of crater
#     store_crater_sorted_idx = np.hstack((np.zeros(max_craters)))

#     for j in range(0, max_craters)[::-1]:
#         print(j)
#         coordinates_center, coordinates_rim, lat_points, lon_points = ellipse_on_ellipsoid(
#             lat_sorted[j], long_sorted[j], major_sorted[j], minor_sorted[j], angle_sorted[j])

#         # plot crater center
#         plt.figure(1)
#         plt.plot(long_plot[j], lat_plot[j],
#                  color='r', marker='o', linestyle='None', markersize=1.0)  # for 2D

#         # conversion from degrees to pixels
#         lat_pointsdeg = lat_points*180/np.pi
#         lon_pointsdeg = lon_points*180/np.pi
#         lon_pointsdeg[np.where(lon_pointsdeg > 180.)] += -360.
#         latpoints_plot = height/180 * (90 - lat_pointsdeg)
#         lonpoints_plot = width/2 * lon_pointsdeg/180 + width/2

#         plt.figure(1)
#         plt.plot(lonpoints_plot, latpoints_plot,
#                  color='r', marker='o', linestyle='None', markersize=1.0)  # for 2D

#         center_x, center_z, select_x, select_z, center_visible, visible = crater_mapping(
#             sim_date, mission_specifics, camera_definition, coordinates_rim, coordinates_center, img_width, img_height, fig3d_idx)

#         if visible > 0:  # if fully or partially visible, store pixel values
#             store_centers[j, :] = [center_x, center_z]
#             store_center_visible[j] = center_visible
#             store_visible[j] = visible  # 0, 1 (partially) or 2 (fully)
#             if center_visible == 1:  # concatenate center coordinates
#                 all_x = np.hstack((select_x, center_x))
#                 all_z = np.hstack((select_z, center_z))
#             else:
#                 all_x = select_x
#                 all_z = select_z
#             store_rim_bounds[j, :] = [np.min(all_x), np.max(
#                 all_x), np.min(all_z), np.max(all_z)]
#             # two rows, as many columns as visible points along the rim
#             store_rims[j] = np.vstack((select_x, select_z))
#             store_crater_sorted_idx[j] = j

#             # plot crater bounds
#             fig3D = plt.figure(fig3d_idx)
#             plt.plot(store_rim_bounds[j, 0]*np.hstack(np.ones(2)), np.hstack((store_rim_bounds[j, 2], store_rim_bounds[j, 3])),
#                      color='blue', linewidth=0.5)  # for 2D
#             plt.plot(store_rim_bounds[j, 1]*np.hstack(np.ones(2)), np.hstack((store_rim_bounds[j, 2], store_rim_bounds[j, 3])),
#                      color='blue', linewidth=0.5)  # for 2D
#             plt.plot(np.hstack((store_rim_bounds[j, 0], store_rim_bounds[j, 1])), store_rim_bounds[j, 2]*np.hstack(np.ones(2)),
#                      color='blue', linewidth=0.5)  # for 2D
#             plt.plot(np.hstack((store_rim_bounds[j, 0], store_rim_bounds[j, 1])), store_rim_bounds[j, 3]*np.hstack(np.ones(2)),
#                      color='blue', linewidth=0.5)  # for 2D

#         else:  # if not visible, remove from stored variables
#             store_centers = np.delete(store_centers, j, axis=0)
#             store_center_visible = np.delete(store_center_visible, j)
#             store_visible = np.delete(store_visible, j)
#             store_rim_bounds = np.delete(store_rim_bounds, j, axis=0)
#             store_rims.pop(j)
#             store_crater_sorted_idx = np.delete(store_crater_sorted_idx, j)
#     fig3D.savefig('./Training/img_3D_{}.png'.format(int(training_idx)),
#                   dpi=500, bbox_inches='tight')
#     np.savez("./Training/data_{}.npz".format(int(training_idx)), training_idx=training_idx, store_centers=store_centers, store_center_visible=store_center_visible,
#              store_visible=store_visible, store_rim_bounds=store_rim_bounds, store_crater_sorted_idx=store_crater_sorted_idx, sim_date=sim_date, camera_definition=camera_definition, mission_specifics=mission_specifics)
# plt.show()
# error('g')
# ###


# --------------------- # Initial date for simulations # --------------------- #
# date = "2022-06-13 10:00"  # You can also use Julian Days: date = "JD2458327.500000"
# nowET = spice.str2et(date+" TDB")  # s past J2000

# # --------------------- # Input S/C position # --------------------- #
# # SCENARIO: 1 = around Moon looking at Moon
# # SCENARIO: 2 = around Earth looking at Moon
# # SCENARIO: 3 = around Earth looking at Earth
# case = 1

# if case == 1:
#     look_at_object = 'MOON'
#     theta = 100*(math.pi/180)
#     dist = 25000
#     sc_xpos = dist*math.cos(theta)
#     sc_ypos = dist*math.sin(theta)
#     sc_zpos = 0
#     moon_sp = spice.spkpos("301", nowET, "J2000", "NONE", "399")
#     moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
#     sc_pos = np.array(moon_pos) + np.array([sc_xpos, sc_ypos, sc_zpos])
# elif case == 2:
#     look_at_object = 'MOON'
#     sc_xpos = 75000
#     sc_ypos = -50000
#     sc_zpos = 0
#     sc_pos = [sc_xpos, sc_ypos, sc_zpos]
# elif case == 3:
#     look_at_object = 'EARTH'
#     theta = 100*(math.pi/180)
#     dist = 100000
#     sc_xpos = dist*math.cos(theta)
#     sc_ypos = dist*math.sin(theta)
#     sc_zpos = 0
#     sc_pos = [sc_xpos, sc_ypos, sc_zpos]

# Camera Settings
# focal_len = 35          # mm
# cam_width = 2048        # pixels
# cam_height = 1536       # pixels
# px = 4.96E-3            # mm of 1 pixel
# camera_definition = [look_at_object, focal_len, cam_width, cam_height, px]
# iter = 0
# reflection = 1
idx_case = 0
[sim_date, camera_definition, sc_pos, iter,
            reflection, mission_specifics] = case_type(2000, training_idx=idx_case, state=[np.pi/4,np.pi/8, 24000], days_past=0.)
nowET = sim_date
# --------------------- # Image Generation # --------------------- #
img_name, earth_map, cloud_map = gen_moon_earth(sc_pos, nowET, camera_definition, iter, reflection)
error('s')
img_orig = cv2.imread(img_name)
img_noise = simulateNoiseColor_scaled(gg_blur(img_orig), factor=1)
cv2.imwrite('img_noise.png', img_noise)
print('---Image has been generated---')

sun_sp = spice.spkpos("10", sim_date, "J2000", "NONE", "399")
sun_pos = [sun_sp[0][0], sun_sp[0][1], sun_sp[0][2]]

moon_sp = spice.spkpos("301", sim_date, "J2000", "NONE", "399")
moon_pos = [moon_sp[0][0], moon_sp[0][1], moon_sp[0][2]]
# # --------------------- # Image Processing # --------------------- #
# # img_try = cv2.imread('Synthetic/50000/moon_8.png')
# img_try = cv2.imread(img_name)

# # Add noise to the image. (https://scikit-image.org/docs/stable/api/skimage.util.html#random-noise)
# noise_img = random_noise(img_try, mode='gaussian', mean=0.045, var=1.4E-4)
# # The above function returns a floating-point image
# # on the range [0, 1], thus we changed it to 'uint8'
# # and from [0,255]
# noise_img = np.array(255*noise_img, dtype='uint8')
# # Display the noise image
# cv2.imwrite('earth_moon_img_noise.png', noise_img)
# # cv2.imshow('blur',noise_img)
# # cv2.waitKey(0)
# img_try = noise_img

# # Kernel Size for Blurring in Edge Detection
# if dist >= 300000:
#     ksize = 7
# elif dist >= 75000:
#     ksize = 15
# elif dist >= 50000:
#     ksize = 25
# elif dist >= 25000:
#     ksize = 35  # 25
# else:
#     ksize = 55  # 35

# # Canny + Edge Removal
focal_len_mm = camera_definition[1]
camera_width = camera_definition[2]
camera_height = camera_definition[3]
px_mm = camera_definition[4]
x_off, y_off = ellipsoid_projection(sc_pos, moon_pos, sim_date, focal_len_mm, px_mm)
R_bi = rot_body2inertial(sc_pos, moon_pos)
R_ib = R_bi.T
print
sun_vec_i = np.vstack(np.array(moon_pos)-np.array(sun_pos))
# edge_pts, edge_coor = true_edge(img_try, R_ib, sun_vec_i, ksize)
edge_pts, edge_coor, edge_pts55, edge_coor55 = sobel55(img_noise, R_ib, sun_vec_i, idx_case)

x_c, y_c, flag = ellipse_fit(edge_coor)
x_c = x_c + x_off
y_c = y_c + y_off

# # Direct Ellipse Fitting
# x_c, y_c, flag = ellipse_fit(edge_coor, img_try)
# x_off, y_off = ellipsoid_projection(
#     sc_pos, moon_pos, nowET, camera_definition, img_name)
# x_c = x_c + x_off
# y_c = y_c + y_off
# print('Estimated center: (' +
#       str(x_c)+', '+str(y_c)+')')

# ang_err_y, ang_err_x, ang_err = centroid_acc(img_try, x_c, y_c, 35)
# print('Angular error of centroiding: ' +
#       str(ang_err*3600*180/pi)+' arcsec')

# # Position Estimation
# range_meanVal, range_stdDev = pos_estimation(
#     x_c, y_c, edge_coor, R_bi, edge_pts)

# # Christian Robinson Algorithm for Position Estimation
r_sc2moon_cam_CR, range_CR = christian_robinson(sc_pos, moon_pos, edge_coor, sim_date, camera_width, camera_height, focal_len_mm, px_mm)
r_sc2moon_estimated = np.dot(R_bi, np.array([r_sc2moon_cam_CR[0], r_sc2moon_cam_CR[2], -r_sc2moon_cam_CR[1]]))
r_sc2moon_true = np.array(moon_pos)-sc_pos
print('Estimated relative position')
print(r_sc2moon_estimated)
print('True relative position')
print(r_sc2moon_true)

# print('done')
# print()
# End of main
spice.kclear()
