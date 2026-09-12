#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#CREATOR
#Tim Kilduff, UC San Diego, July 2023 [UPDATE JUNE 2024]
#tkilduff@ucsd.edu

#MENTORS
#Dr. Pablo Machuca, MIT/SDSU, pmachuca@mit.edu
#Dr. Aaron J. Rosegren, UCSD, ajrosengren@eng.ucsd.edu

# Import all functions from functions.py
from src.functions import *
import random
import sys

#SLURM###############################################################################
# Depends of if this is for training or testing
# training_idx = int(sys.argv[1])  # get task ID
# small_idx = int(sys.argv[2])  # get smallest task ID
# large_idx = int(sys.argv[3])  # get largest task ID
# total_idx = int(sys.argv[4])  # get total number of tasks
# plots_yesno = int(sys.argv[5]) # plots on/off: 1=yes,0=no
# folder_date = sys.argv[6] # date on the folder within ./training
# labels_yesno = int(sys.argv[7]) #generate labels: 1=yes,0=no
# copy_yesno = int(sys.argv[8]) #copy files from prefilter: 1=yes,0=no
# prefilter_date = sys.argv[9] # date on the folder within ./prefilter
# img_range = sys.argv[10] # test image range (oob,in range)
# train_num = sys.argv[11] # number of craters trained
# testdate = sys.argv[12] # date for images (either same as training or new)
# num_craters_train = int(sys.argv[13]) #number of craters used for training
# train_or_test = int(sys.argv[14]) #either for testing (=1) or training (=0)
# blur_yesno = int(sys.argv[15]) # duplicates images with different levels of blur/random noise. 1=yes,0=no
training_idx = 0#int(sys.argv[1])  # get task ID
small_idx = 0#int(sys.argv[2])  # get smallest task ID
large_idx = 1#int(sys.argv[3])  # get largest task ID
total_idx = 2#int(sys.argv[4])  # get total number of tasks
plots_yesno = 1#int(sys.argv[5]) # plots on/off: 1=yes,0=no
folder_date = 2025#sys.argv[6] # date on the folder within ./training
labels_yesno = 1#int(sys.argv[7]) #generate labels: 1=yes,0=no
copy_yesno = 1#int(sys.argv[8]) #copy files from prefilter: 1=yes,0=no
prefilter_date = 2025#sys.argv[9] # date on the folder within ./prefilter
img_range = "inrange"#sys.argv[10] # test image range (oob,in range)
train_num = 1000#sys.argv[11] # number of craters trained
testdate = 2025#sys.argv[12] # date for images (either same as training or new)
num_craters_train = 1000#int(sys.argv[13]) #number of craters used for training
train_or_test = 0#int(sys.argv[14]) #either for testing (=1) or training (=0)
blur_yesno = 1#int(sys.argv[15]) # duplicates images with different levels of blur/random noise. 1=yes,0=no
    
############################################################################
#first, extract various variables
#all_thetaphidist:[theta, phi, dist]
#mission_specifics:[theta, phi, dist, sc_xpos,sc_ypos, sc_zpos, moon_sp, central_pos, sc_pos]
#central_pos
#sun_pos
if train_or_test == 0:    
    data = np.load(f'./prefilter/{prefilter_date}/datafiles/image_gen_allcases.npz') #supercloud
    source_dir = f"./prefilter/{prefilter_date}/images"
    source_dir2 = f"./prefilter/{prefilter_date}/datafiles"    
    destination_dir = f"./training/{folder_date}/images/train2025/povray_images/{train_num}"
    destination_dir2 = f"./training/{folder_date}/datafiles/train2025/npz_original/{train_num}"
else:
    data = np.load(f'./prefilter/{prefilter_date}/{train_num}/{img_range}/{testdate}/datafiles/image_gen_allcases.npz') #supercloud
    source_dir = f"./prefilter/{prefilter_date}/{train_num}/{img_range}/{testdate}/images"
    source_dir2 = f"./prefilter/{prefilter_date}/{train_num}/{img_range}/{testdate}/datafiles"
    destination_dir = f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}"
    destination_dir2 = f"./training/{folder_date}/datafiles/test/npz_original/{train_num}/{img_range}/{testdate}"

lst = data.files
source_all = os.path.join(source_dir2, "image_gen_allcases.npz")
destination_all = os.path.join(destination_dir2, "image_gen_allcases.npz")
shutil.copy2(source_all, destination_all)

#extract contents of image_gen_allcase.npz
dist = np.array(data['all_thetaphidist'][:,2])
# print('len(dist)='+str(len(dist)))
thetas = np.array(data['all_thetaphidist'][:,0])
phi = np.array(data['all_thetaphidist'][:,1])
central_pos = np.array((0,0,0))
# moon_pos = np.array(data['moon_pos'])
camera_definition = list(data['camera_definition'])
reflection = np.array(data['reflection'])
list_months = np.array(data['list_months'])
num_repeat_clouds = float(data['num_repeat_clouds'])

#parallelization
total_cases = len(dist)
vec_cases = np.arange(training_idx, total_cases, total_idx)

# #use theta phi distance -> input into sc_pos
# # spacecraft pos - moon_pos -> sc relative to moon
# # sun_pos - moon_pos -> sun relative to moon
# #dot product -> angle between 45 and 135

#acquire spacecraft position and make it relative to the moon
sc_xpos = dist*np.cos(thetas)*np.cos(phi)
sc_ypos = dist*np.sin(thetas)*np.cos(phi)
sc_zpos = dist*np.sin(phi)
sc_pos = np.transpose([sc_xpos, sc_ypos, sc_zpos])

# pick variables depending on if this is for training or testing
if train_or_test == 0:
    solar_phase = [0,135]
    noiseblur = [1,1.5]
    noise_file_labels = ['_l1','_l15']
    #for training, only one noise level is considered (random between 1 and 1.5)
    min_pix_diam = 16
    visible_condition = 1 #larger than 1, fully visible
    coverage_threshold = 0.4  # visibility threshold for cloud cover
elif train_or_test == 1:
    # solar_phase = [0,180]
    # noiseblur = [0.5,1.125,1.375,2]
    # noise_file_labels = ['_l05','_l125','_l1375','_l2']
    # noise_file_labels = ['_l05','_l125','_l1375','_l2']
    # min_pix_diam = 8
    # visible_condition = 0 #larger than 0, partially and fully visible
    # coverage_threshold = 1.0  # visibility threshold for cloud cover
    solar_phase = [0,180]
    noiseblur = [0.5,1.125,1.375,2]  
    noise_file_labels = ['_l05','_l1125','_l1375','_l2']
    min_pix_diam = 8
    visible_condition = 0 #larger than 0, partially and fully visible
    coverage_threshold = 1.0  # visibility threshold for cloud cover

# Variable for storing image #'s that have no craters
no_crater_detected_imgs = []
   
# # picking colors for plotting
# Viridis
viridis = cm.get_cmap('viridis')
colors = [viridis(0.2), viridis(0.4), viridis(0.6), viridis(0.8)]
# # blues
# blues = cm.get_cmap('Blues')
# colors = [blues(0.8),blues(0.2)]
# Grayscale
# greys = np.linspace(0, 255, 3, dtype=int) #3 different colors for plotting
# colors = [f'#{i:02x}{i:02x}{i:02x}' for i in greys] # array of grays

#plot visible/semi-visible craters for each image
# print('Plotting image, copying files,creating .npz/.txt files')

for idx_month,month in enumerate(list_months):
    for idx_repeat in range(int(num_repeat_clouds)):
        for idx_case in vec_cases:        
            idx_count = (total_cases*num_repeat_clouds)*idx_month+total_cases*idx_repeat+idx_case        
            # if os.path.exists(f"./prefilter/{prefilter_date}/images/earth_img_{int(idx_count)}.png"):                         
            if os.path.exists(os.path.join(source_dir, "earth_img_{}.png".format(int(idx_count)))):            
                source_npz = os.path.join(source_dir2, "image_gen_data_{}.npz".format(int(idx_count)))
                print(img_range)
                print(testdate)
                print('Processing image #'+str(idx_count))  
                data_case = np.load(source_npz) #supercloud
            
                sun_pos = np.array(data_case['sun_pos'])
                #make sun position relative to the moon
                sun_pos_rel_central = sun_pos - central_pos #make into 3x1
                
                sim_date = float(data_case['sim_date'])
                sc_pos_idx = sc_pos[int(idx_case),:]
                
                #do dot product to determine moon angle (we want 45 -> 135 degrees)
                solarphase_angle = np.degrees(np.arccos((np.dot(sc_pos_idx,sun_pos_rel_central))/(np.linalg.norm(sc_pos_idx)*np.linalg.norm(sun_pos_rel_central))))
        
                if solar_phase[0] <= solarphase_angle < solar_phase[1]:

                    formattedindex = f"{int(idx_count):06}"

                    if copy_yesno == 1:
                        #Copy .png files
                        source_png = os.path.join(source_dir, "earth_img_{}.png".format(int(idx_count)))
                        destination_png = os.path.join(destination_dir, "earth_img_{}.png".format(str(formattedindex)))
                        shutil.copy2(source_png, destination_png)

                        # Copy .npz files
                        destination_npz = os.path.join(destination_dir2, "image_gen_data_{}.npz".format(str(formattedindex)))
                        shutil.copy2(source_npz, destination_npz)

                    #Define mission_specifics (currently not defined)
                    mission_specifics = [thetas[int(idx_case)],phi[int(idx_case)],dist[int(idx_case)], sc_xpos[int(idx_case)],sc_ypos[int(idx_case)],sc_zpos[int(idx_case)],
                                            None, central_pos,central_pos + np.array([sc_xpos[int(idx_case)], sc_ypos[int(idx_case)], sc_zpos[int(idx_case)]])]
                    
                    if blur_yesno == 1:
                        if train_or_test == 0:
                            noise_level = (noiseblur[0]+(noiseblur[1]-noiseblur[0])*np.random.random(1)[0])
                            noise_label = noise_file_labels[0]
                            img_distort = Image.fromarray(simulateNoiseColor_scaled(gg_blur(np.array(Image.open(f"./training/{folder_date}/images/train2025/povray_images/{train_num}/earth_img_{str(formattedindex)}.png")),noise_level),1))
                            img_distort.save(f"./training/{folder_date}/images/train2025/povray_images/{train_num}/earth_img_{str(formattedindex)}{noise_label}.png")
                        
                            # for i in range(0,len(noiseblur)):
                            #     img_distort = Image.fromarray(simulateNoiseColor_scaled(gg_blur(np.array(Image.open(f"./training/{folder_date}/images/train2025/povray_images/{train_num}/earth_img_{str(formattedindex)}.png")),noiseblur[i]),1))
                            #     img_distort.save(f"./training/{folder_date}/images/train2025/povray_images/{train_num}/earth_img_{str(formattedindex)}{noise_file_labels[i]}.png")
                        else:
                            # noise_level = (noiseblur[0]+(noiseblur[1]-noiseblur[0])*np.random.random(1)[0])
                            # noise_label = noise_file_labels[0]
                            # img_distort = Image.fromarray(simulateNoiseColor_scaled(gg_blur(np.array(Image.open(f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}/earth_img_{str(formattedindex)}.png")),noise_level),1))
                            # img_distort.save(f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}/earth_img_{str(formattedindex)}{noise_label}.png")

                            for i in range(0,len(noiseblur)):
                                img_distort = Image.fromarray(simulateNoiseColor_scaled(gg_blur(np.array(Image.open(f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}/earth_img_{str(formattedindex)}.png")),noiseblur[i]),1))
                                img_distort.save(f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}/earth_img_{str(formattedindex)}{noise_file_labels[i]}.png")

                    fig3d_idx = idx_count+2

                    #import distance for this instance
                    dist_inst = dist[idx_case]

                    #image overlay
                    if plots_yesno == 1:
                        if train_or_test == 0:
                            img = mpimg.imread(f"./training/{folder_date}/images/train2025/povray_images/{train_num}/earth_img_{formattedindex}.png")
                        else:
                            img = mpimg.imread(f"./training/{folder_date}/images/test/povray_images/{train_num}/{img_range}/{testdate}/earth_img_{str(formattedindex)}.png")
                        FIG_rim = plt.figure(fig3d_idx, figsize=(15, 15))
                        plt.axis('off')
                        plt.imshow(img, cmap=plt.cm.gray)
                        fig3D = plt.figure(fig3d_idx+1, figsize=(15, 15))
                        plt.imshow(img, cmap=plt.cm.gray)
                        plt.axis('off')

                    # if image has different number of pixels than cam_width/height
                    img_width = int(camera_definition[2])
                    img_height = int(camera_definition[3])
                    # focal_length_train = int(camera_definition[1])
                    # px_train = float(camera_definition[4])

                    earth_map_short = data_case['earth_map']
                    cloud_map_short = data_case['cloud_map']
                    earth_map = f"./Maps/months/{earth_map_short}"
                    cloud_map = f"./Maps/clouds/allyear/{cloud_map_short}"

                    coordinates_coastline, center_boxes, filtered_boxes_info, all_boxes, all_filtered_boxes, coverage_list, total_points = coastline_func(earth_map, cloud_map, coverage_threshold)
                    
                    total_boxes = (total_points)*9
                    num_points = filtered_boxes_info[:,0].astype(int)
                    type_box = filtered_boxes_info[:,1]
                    id_box = filtered_boxes_info[:,2].astype(int)
                    id_bigbox = filtered_boxes_info[:,3].astype(int)
                    case_id = filtered_boxes_info[:,8].astype(int)

                    center_x, center_z, plot_rim_x, plot_rim_z, isrim_visible, center_visible, visible, angle_sun_deg, sc_inc_angle_deg = coastline_mapping_vec(sim_date, mission_specifics, camera_definition, coordinates_coastline, center_boxes, fig3d_idx, plots_yesno,
                                train_or_test, num_points)
                    
                    num_boxes = len(num_points)
                    num_pointscraters = np.sum(num_points)
                    cumsum_points = np.cumsum(num_points)

                    if (plots_yesno == 1):
                        FIG_rim = plt.figure(fig3d_idx, figsize=(15, 15))
                        for idx_box in range(len(visible)):         
                            angle_sun_deg_i = angle_sun_deg[idx_box]                              
                            visible_i = visible[idx_box]
                            condition_met = False
                            if (solar_phase[1]>angle_sun_deg_i>=solar_phase[0]) and visible_i>0:
                                condition_met = True
                                    
                            if condition_met:
                                center_visible_i = center_visible[idx_box]                        
                                idx_first = 0 if idx_box == 0 else cumsum_points[idx_box - 1]
                                rim_select = isrim_visible[idx_first:cumsum_points[idx_box]]
                                subset_x = plot_rim_x[idx_first:cumsum_points[idx_box]]
                                subset_z = plot_rim_z[idx_first:cumsum_points[idx_box]]
                                select_x = subset_x[rim_select]
                                select_z = subset_z[rim_select]
                                if (center_visible_i == 1) and (visible_i == 2):  # center is visible and whole rim as well
                                    plt.plot(select_x, select_z, linewidth=0.5, color=colors[2], linestyle='None', marker='.', markersize=1.0)  # for 2D
                                    plt.axis('off')
                                    plt.plot(center_x[idx_box], center_z[idx_box], color=colors[3], marker='x',linestyle='None', markersize=1.0)
                                elif center_visible_i == 1:  # center is visible but not the whole rim
                                    plt.plot(select_x, select_z, linewidth=0.5, color=colors[0], linestyle='None', marker='.', markersize=1.0)  # for 2D
                                    plt.axis('off')
                                    plt.plot(center_x[idx_box], center_z[idx_box], color=colors[1], marker='x',linestyle='None', markersize=1.0)
                        
                    idx_visible = np.where(visible > visible_condition)[0]

                    if len(idx_visible)>0:
                        type_box_all = type_box[idx_visible]
                        case_id_all = case_id[idx_visible]
                        id_box_all = id_box[idx_visible]
                        id_bigbox_all = id_bigbox[idx_visible]
                        coverage_list_all = coverage_list[idx_visible]
                        store_feature_radius_all = type_box_all
                        store_feature_sorted_idx_all = id_box_all
                        
                        store_centers = np.hstack((np.vstack((center_x[idx_visible])),np.vstack((center_z[idx_visible]))))
                        store_visible = visible[idx_visible] #[visible onlyx4]
                        store_angle_sun_deg_all = np.vstack((angle_sun_deg[idx_visible]))
                        sc_inc_angle_deg_all = np.vstack((sc_inc_angle_deg[idx_visible]))

                        # Matrices storage
                        pixel_extent_all = np.zeros((len(idx_visible),1))
                        store_box_bounds_all = np.zeros((len(idx_visible),4))
                        centers_box_all = np.zeros((len(idx_visible),2))
                        widthheight_box_all = np.zeros((len(idx_visible),2))

                        idx_all = -1
                        for idx_box in idx_visible:
                            idx_all = idx_all+1
                            idx_first = 0 if idx_box == 0 else cumsum_points[idx_box - 1]
                            rim_select = isrim_visible[idx_first:cumsum_points[idx_box]]
                            sub_vec_x = plot_rim_x[idx_first:cumsum_points[idx_box]]
                            sub_vec_z = plot_rim_z[idx_first:cumsum_points[idx_box]]
                            
                            notvisible_all = rim_select==0
                            sub_vec_x[notvisible_all] = np.nan
                            sub_vec_z[notvisible_all] = np.nan

                            any_right = np.any(sub_vec_x>=img_width-5e-7*img_width)
                            any_left = np.any(sub_vec_x<=5e-7*img_width)
                            any_above = np.any(sub_vec_z<=5e-7*img_height)
                            any_below = np.any(sub_vec_z>=img_height-5e-7*img_height)

                            if any_right:                    
                                max_x_all = img_width
                            else:
                                max_x_all = np.nanmax(sub_vec_x)
                            if any_left:
                                min_x_all = 0
                            else:
                                min_x_all = np.nanmin(sub_vec_x)
                            if any_below:
                                max_z_all = img_height
                            else:
                                max_z_all = np.nanmax(sub_vec_z)
                            if any_above:
                                min_z_all = 0
                            else:
                                min_z_all = np.nanmin(sub_vec_z)

                            center_box_x = (max_x_all+min_x_all)/2./img_width
                            center_box_z = (max_z_all+min_z_all)/2./img_height

                            width_box_x = (max_x_all-min_x_all)/img_width
                            height_box_z = (max_z_all-min_z_all)/img_height

                            # shift centers and height/width for craters who exit the frame
                            if any_left:
                                center_box_x = center_box_x+5e-7
                            if any_right:                    
                                center_box_x = center_box_x-5e-7
                            if any_above:
                                center_box_z = center_box_z+5e-7
                            if any_below:
                                center_box_z = center_box_z-5e-7

                            store_box_bounds_all[idx_all,:] = [min_x_all, max_x_all, min_z_all, max_z_all]
                            centers_box_all[idx_all,:] = [center_box_x,center_box_z]
                            widthheight_box_all[idx_all,:] = [width_box_x,height_box_z]

                            pixel_extent_all[idx_all] = ((max_x_all-min_x_all)+(max_z_all-min_z_all))/2.

                        is_large_enough = (pixel_extent_all>min_pix_diam).flatten()

                        idx_combined = idx_visible[is_large_enough]
                        if len(idx_combined)>0: #if any visible boxes above pixel threshold 
                            type_box_visible = type_box[idx_combined] 
                            case_id_visible = case_id[idx_combined]                            
                            id_box_visible = id_box[idx_combined]
                            id_bigbox_visible = id_bigbox[idx_combined]                          
                            coverage_list_visible = coverage_list[idx_combined]
                            store_center_visible = np.hstack((np.vstack((center_x[idx_combined])),np.vstack((center_z[idx_combined]))))
                            store_box_bounds = store_box_bounds_all[is_large_enough,:] #only visible
                            # store_box_bounds_all = all_rim_bounds #[this should include all of the boxes regardless of visibility, allx4]
                            store_feature_sorted_idx = id_box_visible
                            store_angle_sun_deg = np.vstack((angle_sun_deg[idx_combined]))
                            store_feature_radius = type_box_visible # include here the id_box=[0,1,2] from MATLAB code, ONLY VISIBLE boxes
                            pixel_extent = pixel_extent_all[is_large_enough]
                            centers_box = centers_box_all[is_large_enough,:]
                            widthheight_box = widthheight_box_all[is_large_enough,:]
                            
                            clabel = 0
                        else: #no boxes above pixel threshold
                            coverage_list_visible = []
                            type_box_visible = []
                            case_id_visible = []
                            id_bigbox_visible = []
                            store_center_visible = []
                            store_box_bounds = []
                            store_feature_sorted_idx = []
                            store_angle_sun_deg = []
                            store_feature_radius = []
                            pixel_extent = []

                            no_crater_detected_imgs.append(str(formattedindex))
                            clabel = 1

                        #plotting all craters
                        if (plots_yesno == 1) and len(store_box_bounds)>0:
                            fig3D = plt.figure(fig3d_idx+1, figsize=(15, 15))
                            plt.axis('off')
                            newbox = np.inf
                            for idx_box in range(len(store_center_visible)):
                                #plot crater bounds                                
                                if visible[idx_combined[idx_box]]==2: #fully visible
                                    # color_box = colors[2]
                                    # color_center = colors[3]
                                    
                                    if newbox!=id_bigbox_visible[idx_box]:
                                        newbox = id_bigbox_visible[idx_box]
                                        red = random.randint(0, 255)/255.
                                        green = random.randint(0, 255)/255.
                                        blue = random.randint(0, 255)/255.
                                    if type_box_visible[idx_box]==0: #full box
                                        color_box = (red,green,blue,1)
                                        color_center = (red,green,blue,1)
                                    elif type_box_visible[idx_box]==1: #half box                                       
                                        color_box = (red,green,blue,0.6)
                                        color_center = (red,green,blue,0.6)
                                    elif type_box_visible[idx_box]==2: #quarter box                                    
                                        color_box = (red,green,blue,0.3)
                                        color_center = (red,green,blue,0.3)
                                else: #partially visible
                                    color_box = colors[0]
                                    color_center = colors[1]
                                plt.plot(store_box_bounds[idx_box, 0]*np.hstack(np.ones(2)), np.hstack((store_box_bounds[idx_box, 2], store_box_bounds[idx_box, 3])),
                                            color=color_box, linewidth=0.5)  # for 2D
                                plt.plot(store_box_bounds[idx_box, 1]*np.hstack(np.ones(2)), np.hstack((store_box_bounds[idx_box, 2], store_box_bounds[idx_box, 3])),
                                            color=color_box, linewidth=0.5)  # for 2D
                                plt.plot(np.hstack((store_box_bounds[idx_box, 0], store_box_bounds[idx_box, 1])), store_box_bounds[idx_box, 2]*np.hstack(np.ones(2)),
                                            color=color_box, linewidth=0.5)  # for 2D
                                plt.plot(np.hstack((store_box_bounds[idx_box, 0], store_box_bounds[idx_box, 1])), store_box_bounds[idx_box, 3]*np.hstack(np.ones(2)),
                                            color=color_box, linewidth=0.5)  # for 2D                                
                                # plt.plot(img_width*centers_box[idx_box,0], img_height*centers_box[idx_box,1], color=color_center, marker='x',linestyle='None', markersize=1.0)
                                plt.annotate(
                                    str(id_bigbox_visible[idx_box])+'.'+str(int(case_id_visible[idx_box])), 
                                    xy=(img_width*centers_box[idx_box,0], img_height*centers_box[idx_box,1]),              # Coordinates of the point to annotate
                                    fontsize=1,             # Text font size
                                    color=color_center[0:3],             # Text color
                                    ha='center',
                                    va='center'
                                )
                    else: #no visible craters
                        no_crater_detected_imgs.append(str(formattedindex))
                        clabel = 1

                        store_centers = []
                        store_center_visible = []
                        store_visible = []
                        store_box_bounds = []
                        store_feature_sorted_idx = []
                        store_feature_sorted_idx_all = []
                        store_box_bounds_all = []
                        store_angle_sun_deg = []
                        store_angle_sun_deg_all = []
                        store_feature_radius = []
                        store_feature_radius_all = []
                        sc_inc_angle_deg_all = []
                        pixel_extent = []
                        pixel_extent_all = []                        
                        coverage_list_all = []
                        coverage_list_visible = []
                        type_box_all = []
                        type_box_visible = []
                        case_id_all = []
                        case_id_visible = []
                        id_bigbox_all = []
                        id_bigbox_visible = []

                    if plots_yesno == 1 and clabel == 0:
                        if train_or_test == 0:
                            FIG_rim.savefig(f'./training/{folder_date}/images/train2025/all_craters/{train_num}/img_{str(formattedindex)}_rim.png',dpi=500, bbox_inches='tight')
                            fig3D.savefig(f'./training/{folder_date}/images/train2025/all_craters/{train_num}/img_{str(formattedindex)}_box.png',dpi=500, bbox_inches='tight')
                        else:
                            FIG_rim.savefig(f'./training/{folder_date}/images/test/all_craters/{train_num}/{img_range}/{testdate}/img_{str(formattedindex)}_rim.png',dpi=500, bbox_inches='tight')
                            fig3D.savefig(f'./training/{folder_date}/images/test/all_craters/{train_num}/{img_range}/{testdate}/img_{str(formattedindex)}_box.png',dpi=500, bbox_inches='tight')
                        plt.close(FIG_rim)
                        plt.close(fig3D)
                    elif plots_yesno == 1 and clabel == 1:
                        plt.close(FIG_rim)
                        plt.close(fig3D)

                    #6RETRIEVE NEW .npz FILE, CREATE .txt FILE, AND PLOT ACTUAL CRATERS#########################################
                    # #.npz -> .txt for yolov5##############################################################
                    if labels_yesno == 1 and clabel == 0:
                        #create a .txt file to input crater's centers and bounds (training)
                        if train_or_test == 0:
                            save_txt_path = f'./training/{folder_date}/labels/train2025/povray_images/{train_num}'
                        else:
                            save_txt_path = f'./training/{folder_date}/labels/test/povray_images/{train_num}/{img_range}/{testdate}'
                            # add the case where there is no noise to the noise_file_labels variable
                            noise_file_labels.append('')

                        for noise in noise_file_labels:
                            noise_label = noise
                            if train_or_test == 0: #training
                                noise_label = noise_file_labels[0]
                            #create a .txt file to input crater's centers and bounds (training)
                            store_crater_info = open(os.path.join(save_txt_path, f'earth_img_{str(formattedindex)}{noise_label}.txt'), "w")

                            for h in range(0,len(store_center_visible)):
                                #for training, we label visible craters as "0" classification
                                identification = str(store_feature_sorted_idx[h])
                                store_crater_info.write(identification + ' ')

                                # we want to store the center of the box (instead of center of crater)
                                store_crater_info.write("{:.6f}".format(centers_box[h,0]) + ' ')
                                store_crater_info.write("{:.6f}".format(centers_box[h,1]) + ' ')

                                #storing the bounds (width and height)
                                store_crater_info.write("{:.6f}".format(widthheight_box[h,0]) + ' ')
                                store_crater_info.write("{:.6f}".format(widthheight_box[h,1]) + '\n')

                            store_crater_info.close()

                    # Disable the VisibleDeprecationWarning for this specific code block
                    with warnings.catch_warnings():
                        # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

                        if train_or_test == 0:
                            save_npz_path = f'./training/{folder_date}/datafiles/train2025/npz2txt/{train_num}/crater_info_{str(formattedindex)}.npz'
                        else:
                            save_npz_path = f'./training/{folder_date}/datafiles/test/npz2txt/{train_num}/{img_range}/{testdate}/crater_info_{str(formattedindex)}.npz'
                        np.savez(save_npz_path,
                            idx_case=np.array(idx_case, dtype=object),
                            idx_count=np.array(formattedindex, dtype=object),
                            store_centers=np.array(store_centers, dtype=object),
                            store_center_visible=np.array(store_center_visible, dtype=object),
                            store_visible=np.array(store_visible, dtype=object),
                            store_box_bounds=np.array(store_box_bounds, dtype=object),
                            store_box_bounds_all=np.array(store_box_bounds_all, dtype=object),
                            store_feature_sorted_idx=np.array(store_feature_sorted_idx, dtype=object),
                            store_feature_sorted_idx_all=np.array(store_feature_sorted_idx_all, dtype=object),
                            sim_date=np.array(sim_date, dtype=object),
                            camera_definition=np.array(camera_definition, dtype=object),
                            mission_specifics=np.array(mission_specifics, dtype=object),
                            store_angle_sun_deg=np.array(store_angle_sun_deg, dtype=object),
                            store_angle_sun_deg_all=np.array(store_angle_sun_deg_all, dtype=object),
                            store_feature_radius=np.array(store_feature_radius, dtype=object),
                            store_feature_radius_all=np.array(store_feature_radius_all, dtype=object),
                            store_sc_inc_angle_deg_all=np.array(sc_inc_angle_deg_all, dtype=object),
                            pixel_extent=np.array(pixel_extent, dtype=object),
                            pixel_extent_all=np.array(pixel_extent_all, dtype=object),
                            coverage_list_visible=np.array(coverage_list_visible, dtype=object),
                            coverage_list_all=np.array(coverage_list_all, dtype=object),
                            id_bigbox_visible=np.array(id_bigbox_visible, dtype=object),
                            id_bigbox_all=np.array(id_bigbox_all, dtype=object),
                            case_id_visible=np.array(case_id_visible, dtype=object),
                            case_id_all=np.array(case_id_all, dtype=object))
                        print('file saved:'+str(idx_case))

                    # Re-enable all warnings after the block
                    warnings.resetwarnings()
            #             print('file saved:'+str(idx_case))

if train_or_test==0:
    training_yaml = open('./training_earth.yaml', "w")
    training_yaml.write('path: ./training\n')
    training_yaml.write('train: ./{}/images/train2025/povray_images/{}\n'.format(folder_date,train_num))                    
    training_yaml.write('val: ./{}/images/validation/povray_images/{}\n'.format(folder_date,train_num))                 
    training_yaml.write('names:\n')      
    for idx_box in range(total_boxes):
        training_yaml.write('  {}: {}\n'.format(idx_box,idx_box))

    training_yaml.close()

# %%
