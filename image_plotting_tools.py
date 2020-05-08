'''
Image plotting tools

Written by Patrick Oakes, Ph.D.
https://patrickoakeslab.com
https://github.com/OakesLab

20/04/15 - v 0.5
'''

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
#import glob as glob
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os
import shutil
import subprocess
import czifile

def normalize_image(image, intensity_minimum, intensity_maximum):
    image[image < intensity_minimum] = intensity_minimum
    image[image > intensity_maximum] = intensity_maximum
    image_norm = (image - intensity_minimum) / (intensity_maximum - intensity_minimum)
    return image_norm 

def gray2rgb(image, colormap):
    image_mapped = (255 * colormap(image)).astype('uint8')
    return image_mapped

def inverted_overlay(image_list):
    combined_images = np.stack(image_list)
    overlay = np.mean(combined_images, axis=0).astype('uint8')
    return overlay

def traditional_overlay(image_list):
    combined_images = np.stack(image_list)
    overlay = np.mean(combined_images, axis=0).astype('uint8')
    return overlay

def make_colormaps():
    # Some of these are adapted from Christophe Leterrier
    # https://github.com/cleterrier/ChrisLUTs
    
    # we make a blank holder for making the rest
    cmp_blank = np.zeros((256,3))
    # make different columns to fill with
    cmp_ones_col = np.ones((1,256))
    cmp_zeros_col = np.zeros((1,256))
    cmp_ascending_col = np.linspace(0,1,256)
    cmp_descending_col = np.flipud(cmp_ascending_col)
    
    # iGreys is all the same in each column
    igrey_cmp = cmp_blank.copy()
    igrey_cmp[:,0] = cmp_descending_col
    igrey_cmp[:,1] = cmp_descending_col
    igrey_cmp[:,2] = cmp_descending_col
    # make the grey at the same time
    grey_cmp = np.flipud(igrey_cmp)
    # convert arrays to colormaps
    igrey_cmp = ListedColormap(igrey_cmp, name='iGrey')
    grey_cmp = ListedColormap(grey_cmp, name='Grey')
    
    # for our main colors we fill a single column
    iblue_cmp = cmp_blank.copy()
    iblue_cmp[:,0] = cmp_descending_col
    iblue_cmp[:,1] = cmp_descending_col
    iblue_cmp[:,2] = cmp_ones_col
    iblue_cmp = ListedColormap(iblue_cmp, name='iBlue')
    
    ired_cmp = cmp_blank.copy()
    ired_cmp[:,0] = cmp_ones_col
    ired_cmp[:,1] = cmp_descending_col
    ired_cmp[:,2] = cmp_descending_col
    ired_cmp = ListedColormap(ired_cmp, name='iRed')
    
    igreen_cmp = cmp_blank.copy()
    igreen_cmp[:,0] = cmp_descending_col
    igreen_cmp[:,1] = cmp_ones_col
    igreen_cmp[:,2] = cmp_descending_col
    igreen_cmp = ListedColormap(igreen_cmp, name='iGreen')
    
    imagenta_cmp = cmp_blank.copy()
    imagenta_cmp[:,0] = cmp_ones_col
    imagenta_cmp[:,1] = cmp_descending_col
    imagenta_cmp[:,2] = cmp_ones_col
    imagenta_cmp = ListedColormap(imagenta_cmp, name='iMagenta')
    
    iyellow_cmp = cmp_blank.copy()
    iyellow_cmp[:,0] = cmp_ones_col
    iyellow_cmp[:,1] = cmp_ones_col
    iyellow_cmp[:,2] = cmp_descending_col
    iyellow_cmp = ListedColormap(iyellow_cmp, name='iYellow')
    
    icyan_cmp = cmp_blank.copy()
    icyan_cmp[:,0] = cmp_descending_col
    icyan_cmp[:,1] = cmp_ones_col
    icyan_cmp[:,2] = cmp_ones_col
    icyan_cmp = ListedColormap(icyan_cmp, name='iCyan')
    
    ibordeaux_cmp = cmp_blank.copy()
    ibordeaux_cmp[:,0] = np.linspace(1,204/255,256)
    ibordeaux_cmp[:,1] = cmp_descending_col
    ibordeaux_cmp[:,2] = np.linspace(1,51/255,256)
    ibordeaux_cmp = ListedColormap(ibordeaux_cmp, name='iBordeaux')
    
    ipurple_cmp = cmp_blank.copy()
    ipurple_cmp[:,0] = np.linspace(1,204/255,256)
    ipurple_cmp[:,1] = cmp_descending_col
    ipurple_cmp[:,2] = np.linspace(1,204/255,256)
    ipurple_cmp = ListedColormap(ipurple_cmp, name='iPurple')
    
    iforest_cmp = cmp_blank.copy()
    iforest_cmp[:,0] = cmp_descending_col
    iforest_cmp[:,1] = np.linspace(1,153/255,256)
    iforest_cmp[:,2] = cmp_descending_col
    iforest_cmp = ListedColormap(iforest_cmp, name='iForest')
    
    iorangeBOP_cmp = cmp_blank.copy()
    iorangeBOP_cmp[:,0] = np.linspace(1,248/255,256)
    iorangeBOP_cmp[:,1] = np.linspace(1,173/255,256)
    iorangeBOP_cmp[:,2] = np.linspace(1,32/255,256)
    iorangeBOP_cmp = ListedColormap(iorangeBOP_cmp, name='iOrangeBOP')
    
    iblueBOP_cmp = cmp_blank.copy()
    iblueBOP_cmp[:,0] = np.linspace(1,32/255,256)
    iblueBOP_cmp[:,1] = np.linspace(1,173/255,256)
    iblueBOP_cmp[:,2] = np.linspace(1,248/255,256)
    iblueBOP_cmp = ListedColormap(iblueBOP_cmp, name='iBlueBOP')
    
    ipurpleBOP_cmp = cmp_blank.copy()
    ipurpleBOP_cmp[:,0] = np.linspace(1,148/255,256)
    ipurpleBOP_cmp[:,1] = np.linspace(1,32/255,256)
    ipurpleBOP_cmp[:,2] = np.linspace(1,148/255,256)
    ipurpleBOP_cmp = ListedColormap(ipurpleBOP_cmp, name='iPurpleBOP')
    
    # for our main colors we fill a single column
    blue_cmp = cmp_blank.copy()
    blue_cmp[:,0] = cmp_zeros_col
    blue_cmp[:,1] = cmp_zeros_col
    blue_cmp[:,2] = cmp_ascending_col
    blue_cmp = ListedColormap(blue_cmp, name='Blue')
    
    red_cmp = cmp_blank.copy()
    red_cmp[:,0] = cmp_ascending_col
    red_cmp[:,1] = cmp_zeros_col
    red_cmp[:,2] = cmp_zeros_col
    red_cmp = ListedColormap(red_cmp, name='Red')
    
    green_cmp = cmp_blank.copy()
    green_cmp[:,0] = cmp_zeros_col
    green_cmp[:,1] = cmp_ascending_col
    green_cmp[:,2] = cmp_zeros_col
    green_cmp = ListedColormap(green_cmp, name='Green')
    
    magenta_cmp = cmp_blank.copy()
    magenta_cmp[:,0] = cmp_ascending_col
    magenta_cmp[:,1] = cmp_zeros_col
    magenta_cmp[:,2] = cmp_ascending_col
    magenta_cmp = ListedColormap(magenta_cmp, name='Magenta')
    
    yellow_cmp = cmp_blank.copy()
    yellow_cmp[:,0] = cmp_ascending_col
    yellow_cmp[:,1] = cmp_ascending_col
    yellow_cmp[:,2] = cmp_zeros_col
    yellow_cmp = ListedColormap(yellow_cmp, name='Yellow')
    
    cyan_cmp = cmp_blank.copy()
    cyan_cmp[:,0] = cmp_zeros_col
    cyan_cmp[:,1] = cmp_ascending_col
    cyan_cmp[:,2] = cmp_ascending_col
    cyan_cmp = ListedColormap(cyan_cmp, name='Cyan')
    
    bordeaux_cmp = cmp_blank.copy()
    bordeaux_cmp[:,0] = np.linspace(0,204/255,256)
    bordeaux_cmp[:,1] = cmp_zeros_col
    bordeaux_cmp[:,2] = np.linspace(0,51/255,256)
    bordeaux_cmp = ListedColormap(bordeaux_cmp, name='Bordeaux')
    
    purple_cmp = cmp_blank.copy()
    purple_cmp[:,0] = np.linspace(0,204/255,256)
    purple_cmp[:,1] = cmp_zeros_col
    purple_cmp[:,2] = np.linspace(0,204/255,256)
    purple_cmp = ListedColormap(purple_cmp, name='Purple')
    
    forest_cmp = cmp_blank.copy()
    forest_cmp[:,0] = cmp_zeros_col
    forest_cmp[:,1] = np.linspace(0,153/255,256)
    forest_cmp[:,2] = cmp_zeros_col
    forest_cmp = ListedColormap(forest_cmp, name='Forest')
    
    orangeBOP_cmp = cmp_blank.copy()
    orangeBOP_cmp[:,0] = np.linspace(0,248/255,256)
    orangeBOP_cmp[:,1] = np.linspace(0,173/255,256)
    orangeBOP_cmp[:,2] = np.linspace(0,32/255,256)
    orangeBOP_cmp = ListedColormap(orangeBOP_cmp, name='OrangeBOP')
    
    blueBOP_cmp = cmp_blank.copy()
    blueBOP_cmp[:,0] = np.linspace(0,32/255,256)
    blueBOP_cmp[:,1] = np.linspace(0,173/255,256)
    blueBOP_cmp[:,2] = np.linspace(0,248/255,256)
    blueBOP_cmp = ListedColormap(blueBOP_cmp, name='BlueBOP')
    
    purpleBOP_cmp = cmp_blank.copy()
    purpleBOP_cmp[:,0] = np.linspace(0,148/255,256)
    purpleBOP_cmp[:,1] = np.linspace(0,32/255,256)
    purpleBOP_cmp[:,2] = np.linspace(0,148/255,256)
    purpleBOP_cmp = ListedColormap(purpleBOP_cmp, name='PurpleBOP')

    # from XKCD
    bright_purple_cmp = cmp_blank.copy()
    bright_purple_cmp[:,0] = np.linspace(0,190/255,256)
    bright_purple_cmp[:,1] = np.linspace(0,3/255,256)
    bright_purple_cmp[:,2] = np.linspace(0,253/255,256)
    bright_purple_cmp = ListedColormap(bright_purple_cmp, name='Bright Purple')

    ibright_purple_cmp = cmp_blank.copy()
    ibright_purple_cmp[:,0] = np.linspace(1,190/255,256)
    ibright_purple_cmp[:,1] = np.linspace(1,3/255,256)
    ibright_purple_cmp[:,2] = np.linspace(1,253/255,256)
    ibright_purple_cmp = ListedColormap(ibright_purple_cmp, name='iBright Purple')

    light_blue_cmp = cmp_blank.copy()
    light_blue_cmp[:,0] = np.linspace(0,149/255,256)
    light_blue_cmp[:,1] = np.linspace(0,208/255,256)
    light_blue_cmp[:,2] = np.linspace(0,252/255,256)
    light_blue_cmp = ListedColormap(light_blue_cmp, name='Light Blue')

    ilight_blue_cmp = cmp_blank.copy()
    ilight_blue_cmp[:,0] = np.linspace(1,149/255,256)
    ilight_blue_cmp[:,1] = np.linspace(1,208/255,256)
    ilight_blue_cmp[:,2] = np.linspace(1,252/255,256)
    ilight_blue_cmp = ListedColormap(ilight_blue_cmp, name='iLight Blue')
    
    cmap_dict ={
        'Red' : red_cmp,
        'Green' : green_cmp,
        'Blue' : blue_cmp,
        'Cyan' : cyan_cmp,
        'Magenta' : magenta_cmp,
        'Yellow' : yellow_cmp,
        'Grey' : grey_cmp,
        'Bordeaux' : bordeaux_cmp,
        'Purple' : purple_cmp,
        'Forest' : forest_cmp,
        'BlueBOP' : blueBOP_cmp,
        'OrangeBOP' : orangeBOP_cmp,
        'PurpleBOP' : purpleBOP_cmp,
        'Bright Purple' : bright_purple_cmp,
        'Light Blue' : light_blue_cmp,
        'iRed' : ired_cmp,
        'iGreen' : igreen_cmp,
        'iBlue' : iblue_cmp,
        'iCyan' : icyan_cmp,
        'iMagenta' : imagenta_cmp,
        'iYellow' : iyellow_cmp,
        'iGrey' : igrey_cmp,
        'iBordeaux' : ibordeaux_cmp,
        'iPurple' : ipurple_cmp,
        'iForest' : iforest_cmp,
        'iBlueBOP' : iblueBOP_cmp,
        'iOrangeBOP' : iorangeBOP_cmp,
        'iPurpleBOP' : ipurpleBOP_cmp,
        'iBright Purple' : ibright_purple_cmp,
        'iLight Blue' : ilight_blue_cmp
    }
    
    return cmap_dict

def show_default_colormaps():
    # make a figure of standard colormaps that we can use
    
    # load the colormaps dictionary
    cmap_dict = make_colormaps()

    # start by making a gradient
    gradient = np.linspace(0, 1, 256)
    # stack two of them together to make a 2D matrix that can be used as an image
    gradient = np.vstack((gradient, gradient))

    # make the figure
    colormap_fig, colormap_axes = plt.subplots(nrows=len(cmap_dict))
    # position the figure to accomodate labels
    colormap_fig.subplots_adjust(top=0.9, bottom=0.0, left=0.2, right=0.99)
    # set the title
    colormap_axes[0].set_title('Colormaps', fontsize=14)
    # for loop to plot each colormap in the dictionary
    for ax, key in zip(colormap_axes, cmap_dict):
        # plot the colormap
        ax.imshow(gradient, aspect='auto', cmap=cmap_dict[key])
        # get the position of the axes to position the text correctly
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        # label each axes
        colormap_fig.text(x_text, y_text, cmap_dict[key].name, va='center', ha='right', fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in colormap_axes:
            ax.set_axis_off()
    # show the figure
    colormap_fig.show()

    return

def make_colormap(cmap_input, color_name, show_plot = False):

    # if you need inspiration: https://xkcd.com/color/rgb/
    
    if type(cmap_input) == tuple:
        if len(cmap_input) == 3:
            RGB_values = cmap_input
        else:
            raise Exception("Only tuples of length 3 are valid (e.g. (R,G,B) ) ")
    elif type(cmap_input) == str:
        hex_string = cmap_input
        # remove the # from the hex string if it's there
        hex_string = hex_string.lstrip('#')
    
        # convert to RGB
        RGB_values = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
    else:
        raise Exception("Not a valid colomap input format: Use either Hex string or RGB tuple")
    
        
    
    # make different columns to fill with
    cmp_ones_col = np.ones((1,256))
    cmp_zeros_col = np.zeros((1,256))
    cmp_ascending_col = np.linspace(0,1,256)
    cmp_descending_col = np.flipud(cmp_ascending_col)
    
    # we make a blank array for making the colormap
    cmp_new = np.zeros((256,3))
    cmp_new[:,0] = np.linspace(0,RGB_values[0]/255,256)
    cmp_new[:,1] = np.linspace(0,RGB_values[1]/255,256)
    cmp_new[:,2] = np.linspace(0,RGB_values[2]/255,256)
    cmp_new = ListedColormap(cmp_new, name=color_name)
    
    # make the inverted one
    icmp_new = np.zeros((256,3))
    icmp_new[:,0] = np.linspace(1,RGB_values[0]/255,256)
    icmp_new[:,1] = np.linspace(1,RGB_values[1]/255,256)
    icmp_new[:,2] = np.linspace(1,RGB_values[2]/255,256)
    icmp_new = ListedColormap(icmp_new, name=('i' + color_name))
    
    # make test figure
    gradient = np.linspace(0, 1, 2**8)
    gradient = np.vstack((gradient, gradient))
                              
    if show_plot == True:
        new_cmap_fig, new_cmap_ax = plt.subplots(nrows=2)
        new_cmap_ax[0].imshow(gradient, aspect='auto', cmap=cmp_new)
        new_cmap_ax[0].axis('off')
        new_cmap_ax[0].set_title(cmp_new.name)
        new_cmap_ax[1].imshow(gradient, aspect='auto', cmap=icmp_new)
        new_cmap_ax[1].axis('off')
        new_cmap_ax[1].set_title(icmp_new.name)
        new_cmap_fig.show()
    
    return cmp_new, icmp_new

def image_min_max(images,cmaps,img_min = 0 ,img_max = 0):

    # if no img_min given set up a list of the appropriate length to fill later
    if img_min == 0:
        img_min = np.zeros(len(cmaps))

    # if no img_max given set up a list of the appropriate length to fill later
    if img_max == 0:
        img_max = np.zeros(len(cmaps))
        
    # generate a dictionary of default colormaps
    cmap_dict = make_colormaps()
    for count, colormap in enumerate(cmaps):
        # if the colormap is a string pull it from the dictionary
        if type(colormap) == str:
            # is the string in the default dictionary?
            if colormap in cmap_dict.keys():
                cmaps[count] = cmap_dict[colormap]
            else:
                raise Exception('Not a recognized colormap')


    # create an empty list to hold our image details
    image_list = []

    # loop through each channel provided
    for count, image in enumerate(images):
        # if filenames were given
        if type(image[count]) == str:
            # if it's a string check if it's a czi or tif file
            if image[-3:] == 'tif':
                # read in tif file
                imstack = io.imread(image)
                #check if a minimum intensity was defined - if not set to image minimum
                if img_min[count] == 0.0:
                    img_min[count] = np.min(imstack)
                elif img_min[count] <= 1:
                    # if a value less than 1 is defined use as a percentage
                    if len(imstack.shape) == 2:
                        image_intensities = sorted(imstack.ravel())
                    else:
                        image_intensities = sorted(np.mean(imstack, axis = 0).ravel())
                    N_pixels = len(image_intensities)
                    img_min[count] = int(image_intensities[int(img_min[count] * N_pixels)])

                #check if a maximum intensity was defined - if not set to image maximum
                if img_max[count] == 0.0:
                    img_max[count] = np.max(imstack)
                elif img_max[count] <= 1:
                    # if a value less than 1 is defined use as a percentage
                    if len(imstack.shape) == 2:
                        image_intensities = sorted(imstack.ravel())
                    else:
                        image_intensities = sorted(np.mean(imstack, axis = 0).ravel())
                    N_pixels = len(image_intensities)
                    img_max[count] = int(image_intensities[int(img_max[count] * N_pixels)])

                image_list.append((imstack,cmaps[count],img_min[count],img_max[count]))

            elif image[-3:] == 'czi':
                imstack = czifile.imread(image)
                n_channels = imstack.shape[2]
                n_timepoints = imstack.shape[3]

                for channel in np.arange(0,n_channels):
                    if n_timepoints == 1:
                        channel_stack = imstack[0,0,channel,0,0,:,:,0]
                    else:
                        channel_stack = imstack[0,0,channel,:,0,:,:,0]
                    if img_min[channel] == 0.0:
                        img_min[channel] = np.min(channel_stack)
                    elif img_min[channel] <= 1:
                        if len(channel_stack.shape) == 2:
                            image_intensities = sorted(channel_stack.ravel())
                        else:
                            image_intensities = sorted(np.mean(channel_stack, axis = 0).ravel())
                        N_pixels = len(image_intensities)
                        img_min[channel] = int(image_intensities[int(img_min[channel] * N_pixels)])
                    if img_max[channel] == 0.0:
                        img_max[channel] = np.max(channel_stack)
                    elif img_max[channel] <= 1:
                        if len(channel_stack.shape) == 2:
                            image_intensities = sorted(channel_stack.ravel())
                        else:
                            image_intensities = sorted(np.mean(channel_stack, axis = 0).ravel())
                        N_pixels = len(image_intensities)
                        img_max[channel] = int(image_intensities[int(img_max[channel] * N_pixels)])
                    image_list.append((channel_stack,cmaps[channel],img_min[channel],img_max[channel]))


        else:
            # if it's an array already
            imstack = image
            # if the minimum isn't defined set it equal to the image minimum
            if img_min[count] == 0.0:
                img_min[count] = np.min(image)   
            elif img_min[count] <= 1:
                # if the minimum is defined as less than 1 use it as a percentage
                image_intensities = sorted(image.ravel())
                N_pixels = len(image_intensities)
                img_min[count] = int(image_intensities[int(img_min[count] * N_pixels)])

            # if the maximum isn't defined set it equal to the image maximum
            if img_max[count] == 0.0:
                img_max[count] = np.max(image)
            elif img_max[count] <= 1:
                # if the minimum is defined as less than 1 use it as a percentage
                image_intensities = sorted(image.ravel())
                N_pixels = len(image_intensities)
                img_max[count] = int(image_intensities[int(img_max[count] * N_pixels)])

            image_list.append((imstack,cmaps[count],img_min[count],img_max[count]))

    return image_list

def overlay_image(images, cmps, img_min = 0, img_max = 0):

    image_list = image_min_max(images, cmps, img_min, img_max)
    # create an empty list to hold the channel data
    channels = []
    channels_mapped = []
    
    # for each channel, normalize image and get min and max intensities
    for channel_data in image_list:
        # add it the the channel list
        channels.append(channel_data)

        # check if it's a multiplane image or a single image
        if len(channel_data[0].shape) == 2:
            # normalize the image
            channel_norm = normalize_image(channel_data[0], channel_data[2], channel_data[3])
            # map the normalized image to the colormap indicated
            channel_mapped = gray2rgb(channel_norm, channel_data[1])
        else:
            # get the number of frames
            N_planes = channel_data[0].shape[0]
            channel_mapped = np.zeros((channel_data[0].shape[0], channel_data[0].shape[1], 
                               channel_data[0].shape[2],4), dtype='uint8')
            for plane in np.arange(0,N_planes):
                # normalize the image
                channel_norm = normalize_image(channel_data[0][plane], channel_data[2], channel_data[3])
                # map the normalized image to the colormap indicated
                channel_mapped[plane] = gray2rgb(channel_norm, channel_data[1])

        # add this mapped image to our channel_list variable
        channels_mapped.append(channel_mapped)

    
    # check whether we're using inverted or traditional overlays
    if channel_data[1].name[0] == 'i':
        overlay = inverted_overlay(channels_mapped)
    else:
        overlay = traditional_overlay(channels_mapped)
    
    # determine how many columns to plot
    N_cols = len(image_list)

    if len(channels[0][0].shape) == 2:
        # make the channel figure - plot each channel separately
        channel_fig, channel_axes = plt.subplots(nrows=1, ncols=N_cols)
        # cycle through each channel and plot it using the relative min and max values
        for i in np.arange(0,N_cols):
            channel_axes[i].imshow(channels[i][0], cmap=channels[i][1],vmin=channels[i][2], vmax=channels[i][3])
            channel_axes[i].set_title('Channel ' + str(i + 1) + '\n Min: ' + str(channels[i][2]) + '  Max: ' + str(channels[i][3]))
            channel_axes[i].axis('off')
    
        # display the figure
        channel_fig.tight_layout()
        channel_fig.show()

        # overlay figure
        overlay_fig, overlay_ax = plt.subplots()
        overlay_ax.imshow(overlay)
        overlay_ax.axis('off')
        overlay_ax.set_title('Overlay')
        overlay_fig.show()
    else:
        # make the timelapse figure - plot each channel separately
        channel_fig, channel_axes = plt.subplots(nrows=2, ncols=N_cols)
        # cycle through each channel and plot it using the relative min and max values
        if N_cols > 1:
            for i in np.arange(0,N_cols):
                channel_axes[0,i].imshow(channels[i][0][0], cmap=channels[i][1],vmin=channels[i][2], vmax=channels[i][3])
                channel_axes[0,i].set_title('Channel ' + str(i + 1) + '\nt=0\n Min: ' + str(channels[i][2]) + '  Max: ' + str(channels[i][3]))
                channel_axes[0,i].axis('off')
                channel_axes[1,i].imshow(channels[i][0][-1], cmap=channels[i][1],vmin=channels[i][2], vmax=channels[i][3])
                channel_axes[1,i].set_title('t='+str(N_planes))
                channel_axes[1,i].axis('off')
        else:
            channel_axes[0].imshow(channels[0][0][0], cmap=channels[0][1],vmin=channels[0][2], vmax=channels[0][3])
            channel_axes[0].set_title('Channel ' + str(0 + 1) + '\nt=0\n Min: ' + str(channels[0][2]) + '  Max: ' + str(channels[0][3]))
            channel_axes[0].axis('off')
            channel_axes[1].imshow(channels[0][0][-1], cmap=channels[0][1],vmin=channels[0][2], vmax=channels[0][3])
            channel_axes[1].set_title('t='+str(N_planes))
            channel_axes[1].axis('off')
        
        # display the figure
        channel_fig.tight_layout()
        channel_fig.show()

        # overlay figure
        overlay_fig, overlay_ax = plt.subplots(nrows=1, ncols=2)
        overlay_ax[0].imshow(overlay[0])
        overlay_ax[0].axis('off')
        overlay_ax[0].set_title('Overlay t=0')
        overlay_ax[1].imshow(overlay[-1])
        overlay_ax[1].axis('off')
        overlay_ax[1].set_title('Overlay t='+str(N_planes))
        overlay_fig.show()

    # return the overlay matrix as your output
    return overlay


def make_timestamp_list(frame_interval, N_frames, fmt = 'sec'):
    # construct a list of time points as integers
    time_pts = np.arange(0,frame_interval * N_frames, frame_interval)
    
    # if it's just seconds, add an 's' to the label
    if fmt == 'sec':
        timestamp_list = [(str(i) + ' s') for i in time_pts]
    elif fmt == 'min:sec':
        # if it's in min:sec format create an empty list
        timestamp_list = []
        for t in time_pts:
            # for each time point take the remainder of dividng by 60
            minutes, seconds = divmod(t, 60)
            # add the time point in the right format tot he list
            timestamp_list.append("%02d:%02d" % (minutes, seconds))
    elif fmt == 'hr:min:sec':
        # create an empty list like above
        timestamp_list = []
        for t in time_pts:
            # find the seconds first using the remainder
            minutes, seconds = divmod(t, 60)
            # find the minutes next using the remainder
            hours, minutes = divmod(minutes, 60)
            # format the string correctly
            timestamp_list.append("%02d:%02d:%02d" % (hours, minutes, seconds))
    
    # return the list of timestamps strings
    return timestamp_list

def ffmpeg_str(filesavename, imgstack,**kwargs):

    # get the dimensions of the image
    if len(imgstack.shape) == 3:
        img_h, img_w = imgstack.shape[0:2]
    else:
        img_h, img_w = imgstack.shape[1:3]

    # check if there is a scalebar_length given
    # if so - contstruct a scalebar string to pass to ffmpeg
    if 'scalebar_length' in kwargs:
        width = kwargs['scalebar_length']
        # check if a height of the scalebar is defined
        if 'scaleheight' in kwargs:
            scalebar_height = str(kwargs['scaleheight'])
        else:
            scalebar_height = str(int(img_h * 0.015))

        # check if it has a color defined
        if 'scalecolor' in kwargs:
            scalebar_color = kwargs['scalecolor']
        else:
            scalebar_color = 'black'

        # check if a position has been defined
        if 'scaleposition' in kwargs:
            if type(kwargs['scaleposition']) == str:
                if kwargs['scaleposition'] == 'top_left':
                    scalebar_r = '20'
                    scalebar_c = '20'
                elif kwargs['scaleposition'] == 'bottom_left':
                    scalebar_r = str(img_h - int(scalebar_height) - 20)
                    scalebar_c = '20'
                elif kwargs['scaleposition'] == 'top_right':
                    scalebar_r = '20'
                    scalebar_c = str(img_w - width - 20)
                elif kwargs['scaleposition'] == 'bottom_right':
                    scalebar_r = str(img_h - 20)
                    scalebar_c = str(img_w - width - 20)
            else:
                scalebar_r = str(kwargs['scaleposition'][0])
                scalebar_c = str(kwargs['scaleposition'][1])
        else:
            kwargs['scaleposition'] = 'bottom_right'
            scalebar_r = str(img_h - 20)
            scalebar_c = str(img_w - width - 20)

        # construct the ffmpeg string for the scalebar
        bar_string = '"drawbox=x=' + scalebar_c + ':y=' + scalebar_r + ':w=' + str(width) \
                     + ':h=' + scalebar_height + ':color=' + scalebar_color + '@1.0:t=fill"'

        # check if it has a label
        if 'scalelabel' in kwargs:
            scalebar_label = kwargs['scalelabel']
            scalebar_label_color = scalebar_color

            # check if it has a fontsize
            if 'scalefontsize' in kwargs:
                scalebar_fontsize = str(kwargs['scalefontsize'])
            else:
                scalebar_fontsize = str(int(width / 3))

            # check if it has a position
            if 'scalefontposition' in kwargs:
                scalebar_font_r = str(kwargs['scalefontposition'][0])
                scalebar_font_c = str(kwargs['scalefontposition'][1])        
            else:
                if (kwargs['scaleposition'] == 'bottom_left') or (kwargs['scaleposition'] == 'bottom_right'):
                    scalebar_font_r = str(int(scalebar_r) - int(scalebar_fontsize) - 5)
                    scalebar_font_c = scalebar_c
                else:
                    scalebar_font_r = str(int(scalebar_r) + int(scalebar_height) + 5)
                    scalebar_font_c = scalebar_c

            # construc the ffmpeg string for the scalebar label
            label_string = ',"drawtext=fontsize=' + scalebar_fontsize + ':x=' + scalebar_font_c + \
                           ':y=' + scalebar_font_r + ':fontcolor=' + scalebar_label_color + \
                           ':text=\'' + scalebar_label + '\'"'
        else:
            label_string = ''

        # combine the scalebar and scalebarfont strings into one
        scalebar_string = bar_string + label_string
    else:
        scalebar_string = ''
        

    # check if timestamp strings are given and if so create a timestamp string to pass to ffmpeg
    if 'time_pts' in kwargs:

        # check if a position is given
        if 'timeposition' in kwargs:
            if type(kwargs['timeposition']) == str:
                if kwargs['timeposition'] == 'top_left':
                    timestamp_r = '30'
                    timestamp_c = '30'
                elif kwargs['timeposition'] == 'top_right':
                    timestamp_r = '30'
                    timestamp_c = str(img_w - 3 * int(timestamp_fontsize) - 30)
                elif kwargs['timeposition'] == 'bottom_left':
                    timestamp_r = str(img_h - int(timestamp_fontsize))
                    timestamp_c = '30'
                elif kwargs['timeposition'] == 'bottom_right':
                    timestamp_r = str(img_h - int(timestamp_fontsize))
                    timestamp_c = str(img_w - 3 * int(timestamp_fontsize) - 30)
            else:
                timestamp_r = str(kwargs['timeposition'][0])
                timestamp_c = str(kwargs['timeposition'][1])
        else:
            timestamp_r = '30'
            timestamp_c = str(img_w - 160) 

        # check if a color is given
        if 'timecolor' in kwargs:
            timestamp_color = kwargs['timecolor']
        elif 'scalecolor' in kwargs:
            timestamp_color = scalebar_color
        else:
            timestamp_color = 'black'
        
        # check if a font size is given
        if 'timefontsize' in kwargs:
            timestamp_fontsize = str(kwargs['timefontsize'])
        else:
            timestamp_fontsize = str(int(img_h / 10))

        # the string to grab the text from the metadata
        time_text = '\'%{metadata\:date}\''
    
        # construct the timestamp string
        timestamp_string = '"drawtext=x=' + timestamp_c + ':y=' + timestamp_r + ':fontsize=' \
                      + timestamp_fontsize + ':fontcolor=' + timestamp_color + ':text=' \
                      + time_text + '"'
    else:
        timestamp_string = ''
        

    # combine the scalebar_string and timestamp_string into a single complex filter string for ffmpeg
    if len(scalebar_string) > 0:
        if len(timestamp_string) > 0:
            filter_string = '-filter_complex ' + scalebar_string + ',' + timestamp_string + ' '
        else:
            filter_string = '-filter_complex ' + scalebar_string + ' '
    elif len(timestamp_string) > 0:
        filter_string = '-filter_complex ' + timestamp_string + ' '
    else:
        filter_string = ''

    # check if the format is specified in the input arguments
    if 'fmt' in kwargs:
        # if it's a video output
        if kwargs['fmt'] == 'video':
            # define the codec
            codec = '-vcodec libx264 '
            # define the pixel format for saving
            outfile = '-pix_fmt yuv420p ' + filesavename
            # define the input images
            files = '-i temp_folder/movie%04d.tif '
            # check if a video quality is given
            if 'quality' in kwargs:
                quality = '-crf ' + str(kwargs['quality']) + ' ' 
            else:
                quality = '-crf 25 '
            # check if a framerate is defined
            if 'framerate' in kwargs:
                framerate = '-r ' + str(kwargs['framerate']) + ' '
            else:
                framerate = '-r 10 '
        else:
            # if it's not a video it's a test frame to check layout and most of these
            # parameters aren't needed
            codec = ''
            quality = ''
            framerate = ''
            # change the output file to a single image
            outfile = '-pix_fmt rgba first_frame_overlay.tif'
            # change the input file to the first frame of the movie
            files = '-i first_frame.tif '
    else:
        # if no format is specified give the default video parameters
        # define the codec
        codec = '-vcodec libx264 '
        # define the pixel format for saving
        outfile = '-pix_fmt yuv420p ' + filesavename
        # define the input images
        files = '-i temp_folder/movie%04d.tif '
        # check if a video quality is given
        if 'quality' in kwargs:
            quality = '-crf ' + str(kwargs['quality']) + ' ' 
        else:
            quality = '-crf 25 '
        # check if a framerate is defined
        if 'framerate' in kwargs:
            framerate = '-r ' + str(kwargs['framerate']) + ' '
        else:
            framerate = '-r 10 '

    # define the initial string
    # -y forces overwrite of the file if it exists
    # -f image2 tells it to expect an input file - likely not needed but doesn't hurt
    init_string = 'ffmpeg -y -f image2 '
    
    command_string = init_string + framerate + files \
                    + codec + quality + filter_string + outfile
    
    params = {}
    if 'width' in locals():
        params['scalebar_length'] = int(width)
        params['scaleheight'] = int(scalebar_height)
        params['scalecolor'] = scalebar_color
        params['scaleposition'] = (int(scalebar_r), int(scalebar_c))
    if 'scalelabel' in kwargs:
        params['scalelabel'] = scalebar_label
        params['scalefontsize'] = int(scalebar_fontsize)
        params['scalefontposition'] = (int(scalebar_font_r),int(scalebar_font_c))
    if 'time_pts' in kwargs:
        params['time_pts'] = kwargs['time_pts']
        params['timeposition'] = (int(timestamp_r),int(timestamp_c))
        params['timecolor'] = timestamp_color
        params['timefontsize'] = timestamp_fontsize
    if 'framerate' in kwargs:
        params['framerate'] = kwargs['framerate']
    if 'quality' in kwargs:
        params['quality'] = kwargs['quality']
    
    # return the command string
    return command_string, params

def movie_overlay_test(imstack, **kwargs):
    # check stack shape to make sure it's even or ffmpeg will throw an error
    if imstack.shape[1] % 2:
        imstack = imstack[:,:-1,:]
    if imstack.shape[2] % 2:
        imstack = imstack[:,:,:-1]
    
    # check if a timestamp is included
    if 'time_pts' in kwargs:
        # if so save with metadata tag
        io.imsave('first_frame.tif',imstack[0], plugin='tifffile',
                    extratags=[(306, 's', 0, str(kwargs['time_pts'][0]), True )])
    else:
        # just save normally
        io.imsave('first_frame.tif',imstack[0])
    
    # generate the ffmpeg command
    command_string, params = ffmpeg_str('first_frame_overlay.tif', imstack, fmt = 't', **kwargs)
    
    # run ffmpeg
    subprocess.call(command_string, shell=True)
    
    # delete the first frame
    os.remove('first_frame.tif')
    
    # read in the overlay
    first_frame_overlay = io.imread('first_frame_overlay.tif')
    
    # plot the overlay
    test_fig, test_ax = plt.subplots()
    test_ax.imshow(first_frame_overlay)
    test_fig.show()
    
    for key in params:
        if key != 'time_pts':
            print(key, ' : ', params[key])
    
    return params
    
def save_timelapse_as_movie(savename, imstack, **kwargs):
    
    # make a temp folder to hold the image series
    os.mkdir('temp_folder')
    
    # check stack shape to make sure it's even or ffmpeg will throw an error
    if imstack.shape[1] % 2:
        imstack = imstack[:,:-1,:]
    if imstack.shape[2] % 2:
        imstack = imstack[:,:,:-1]

    # determine the number of images
    N_images = imstack.shape[0]
    
    # write image series
    # if a timestamp is included
    if 'time_pts' in kwargs:
        for plane in np.arange(0,N_images):
            io.imsave('temp_folder/movie%04d.tif' % plane, imstack[plane], plugin='tifffile',
                        extratags=[(306, 's', 0, str(kwargs['time_pts'][plane]), True )])
    else:
        for plane in np.arange(0,N_images):
            io.imsave('temp_folder/movie%04d.tif' % plane, imstack[plane])

    # generate the ffmpeg command
    command_string, params = ffmpeg_str(savename, imstack, **kwargs)
    #print(command_string)
    # run ffmpeg
    subprocess.call(command_string, shell=True)
    
    # delete the temp folder and files
    shutil.rmtree('temp_folder')
    
    return