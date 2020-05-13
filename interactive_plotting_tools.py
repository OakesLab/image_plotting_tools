'''
Interactive Image plotting tools to make image overlays and movies

Written by Patrick Oakes, Ph.D.
https://patrickoakeslab.com
https://github.com/OakesLab

20/05/08
'''

import ipywidgets as widgets                         # widgets
from ipywidgets import fixed, Layout
from image_plotting_tools import *
import os
import matplotlib.pyplot as plt
from IPython.display import display
import glob as glob
import skimage.io as io                                        # reading in images
import csv
import czifile

def overlay_image_widget(images, cmps, img_min = 0, img_max = 0):

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

    # overlay images
    overlay = image_overlay(channels_mapped)

    # return the overlay matrix as your output
    return overlay

# functions to make the channel interactive
def interactive_plot(channel_check, R,G,B, img, t,intensity_minimum, intensity_maximum, c,inverted_check):
    
    if channel_check:
        cmp, icmp = make_colormap((R,G,B),'COLOR',False)

        if inverted_check:
            cmp = icmp
        if img[-3:] == 'tif':
            img = io.imread(img)
        elif img[-3:] == 'czi':
            img = czifile.imread(img)
            img = img[0,0,c,:,0,:,:,0]

        if len(img.shape) > 2:
    #         time.max = img.shape[0]
            fig = plt.figure(figsize=(5,5))
            plt.imshow(img[t], cmap = cmp, vmin = intensity_minimum, vmax = intensity_maximum)
            plt.axis('off')
        else:
            fig = plt.figure(figsize=(5,5))
            plt.imshow(img, cmap = cmp, vmin = intensity_minimum, vmax = intensity_maximum)
            plt.axis('off')
    else:
        fig = plt.figure(figsize=(5,5))
        plt.imshow(np.zeros((100,100)), cmap='Greys')
        plt.text(40,50,'NO IMAGE')
        plt.axis('off')
    return img

def rgb_to_hex(rgb):
    '''converts an rgb tuple to a hex string'''
    return '%02x%02x%02x' % rgb

def overlay_interactive(inverted_check,t,
                        img1_check, img2_check, img3_check,
                        img1,img2,img3,
                        R1,G1,B1,R2,G2,B2,R3,G3,B3,
                        img1_min,img2_min,img3_min,
                        img1_max,img2_max,img3_max):
    file_list, cmap_list, icmap_list, img_min_list, img_max_list = [], [], [], [], []
    if img1_check:
        if img1[-3:] == 'tif':
            image1 = io.imread(img1)
        elif img1[-3:] == 'czi':
            image1 = czifile.imread(img1)
            image1 = image1[0,0,0,:,0,:,:,0]
        if len(image1.shape) > 2:
            image1 = image1[t]
        file_list.append(image1)
        cmp1, icmp1 = make_colormap((R1,G1,B1),'COLOR',False)
        cmap_list.append(cmp1)
        icmap_list.append(icmp1)
        img_min_list.append(img1_min)
        img_max_list.append(img1_max)
    if img2_check:
        if img2[-3:] == 'tif':
            image2 = io.imread(img2)
        elif img2[-3:] == 'czi':
            image2 = czifile.imread(img2)
            image2 = image2[0,0,1,:,0,:,:,0]
        if len(image2.shape) > 2:
            image2 = image2[t]
        
        file_list.append(image2)
        cmp2, icmp2 = make_colormap((R2,G2,B2),'COLOR',False)
        cmap_list.append(cmp2)
        icmap_list.append(icmp2)
        img_min_list.append(img2_min)
        img_max_list.append(img2_max)
    if img3_check:
        if img3[-3:] == 'tif':
            image3 = io.imread(img3)
        elif img3[-3:] == 'czi':
            image3 = czifile.imread(img3)
            image3 = image3[0,0,2,:,0,:,:,0]
        if len(image3.shape) > 2:
            image3 = image3[t]
        
        file_list.append(image3)
        cmp3, icmp3 = make_colormap((R3,G3,B3),'COLOR',False)
        cmap_list.append(cmp3)
        icmap_list.append(icmp3)
        img_min_list.append(img3_min)
        img_max_list.append(img3_max)
    
    if len(file_list) == 0:
        fig = plt.figure(figsize=(5,5))
        plt.imshow(np.zeros((100,100)), cmap='Greys')
        plt.text(40,50,'SELECT A \nCHANNEL')
        plt.axis('off')
    else:
        # create an empty list to hold the channel data
        channels = []
        channels_mapped = []

        if inverted_check:
            image_list = image_min_max(file_list, icmap_list, img_min_list, img_max_list)
        else:
            image_list = image_min_max(file_list, cmap_list, img_min_list, img_max_list)

        # for each channel, normalize image and get min and max intensities
        for channel_data in image_list:
            # add it the the channel list
            channels.append(channel_data)
            # normalize the image
            channel_norm = normalize_image(channel_data[0], channel_data[2], channel_data[3])
            # map the normalized image to the colormap indicated
            channel_mapped = gray2rgb(channel_norm, channel_data[1])
            # add this mapped image to our channel_list variable
            channels_mapped.append(channel_mapped)

        # overlay images
        overlay = image_overlay(channels_mapped)

        if len(overlay.shape) == 3:
            # display the figure
            fig = plt.figure(figsize=(5,5))
            plt.imshow(overlay)
            plt.axis('off')
        else:
            fig = plt.figure(figsize=(5,5))
            plt.imshow(overlay[t])
            plt.axis('off')


def interactive_overlay_images(file_list):

    # dummy parameters to initialize sliders
    img_min=0
    img_max=10000
    
    # load XKCD color dictionaries:
    XKCD_hex2color_dict, XKCD_hex2RGB_dict, XKCD_color2hex_dict, XKCD_color2RGB_dict = load_XKCD_colors()
    
    # widgets for the overlay column
    save_button = widgets.Button(description="Save Image", layout=Layout(width='98%'))
    save_name = widgets.Text(value='', description="Save Name", continuous_update=False)
    inverted_check = widgets.Checkbox(description='Invert Colormaps')
    blank = widgets.Label(value='')
    
    # functions to have update between sliders/colornames/hex colors
    def color_name2RGB_c1(color_name):
        '''sets RGB slider from the color textbox name'''
        (R1.value, G1.value, B1.value) = XKCD_color2RGB_dict[color_name.lower()]
        
    def color_name2RGB_c2(color_name):
        '''sets RGB slider from the color textbox name'''
        (R2.value, G2.value, B2.value) = XKCD_color2RGB_dict[color_name.lower()]
    
    def color_name2RGB_c3(color_name):
        '''sets RGB slider from the color textbox name'''
        (R3.value, G3.value, B3.value) = XKCD_color2RGB_dict[color_name.lower()]

    def hex_to_rgb_c1(hex_string):
        '''sets RGB slider from the hex string textbox'''
        (R1.value, G1.value, B1.value) = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))

    def hex_to_rgb_c2(hex_string):
        '''sets RGB slider from the hex string textbox'''
        (R2.value, G2.value, B2.value) = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
    
    def hex_to_rgb_c3(hex_string):
        '''sets RGB slider from the hex string textbox'''
        (R3.value, G3.value, B3.value) = tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
        
    def color_name_fromRGB_c1(change):
        hex_string = rgb_to_hex((R1.value, G1.value, B1.value))
        color_hex_c1.value = hex_string
        if hex_string in XKCD_hex2RGB_dict:
            color_text_c1.value = XKCD_hex2color_dict[hex_string]
        else:
            color_text_c1.value = ''
            
    def color_name_fromRGB_c2(change):
        hex_string = rgb_to_hex((R2.value, G2.value, B2.value))
        color_hex_c2.value = hex_string
        if hex_string in XKCD_hex2RGB_dict:
            color_text_c2.value = XKCD_hex2color_dict[hex_string]
        else:
            color_text_c2.value = ''
    
    def color_name_fromRGB_c3(change):
        hex_string = rgb_to_hex((R3.value, G3.value, B3.value))
        color_hex_c3.value = hex_string
        if hex_string in XKCD_hex2RGB_dict:
            color_text_c3.value = XKCD_hex2color_dict[hex_string]
        else:
            color_text_c3.value = ''


    # actual widgets channel 1
    channel_check_c1 = widgets.Checkbox(description="Channel 1")
    R1 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'R', continuous_update = False)
    G1 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'G', continuous_update = False)
    B1 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'B', continuous_update = False)
    time_pt = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Time pt', continuous_update = False)
    color_text_c1 = widgets.Text(value='', description="color name", continuous_update = False)
    color_hex_c1 = widgets.Text(value='', description="hex color",continuous_update = False)
    file_name_c1 = widgets.Dropdown(options = file_list, description='File')
    inten_min_c1 = widgets.IntSlider(value = img_min, min = img_min, max = img_max, step = 1, description = 'Min Inten', continuous_update = False)
    inten_max_c1 = widgets.IntSlider(value = img_max, min = img_min, max = img_max, step = 1, description = 'Max Inten', continuous_update = False)

    # actual widgets channel 2
    channel_check_c2 = widgets.Checkbox(description="Channel 2")
    R2 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'R', continuous_update = False)
    G2 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'G', continuous_update = False)
    B2 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'B', continuous_update = False)
#     time_pt_c2 = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Time pt', continuous_update = False)
    color_text_c2 = widgets.Text(value='', description="color name", continuous_update = False)
    color_hex_c2 = widgets.Text(value='', description="hex color",continuous_update = False)
    file_name_c2 = widgets.Dropdown(options = file_list, description='File')
    inten_min_c2 = widgets.IntSlider(value = img_min, min = img_min, max = img_max, step = 1, description = 'Min Inten', continuous_update = False)
    inten_max_c2 = widgets.IntSlider(value = img_max, min = img_min, max = img_max, step = 1, description = 'Max Inten', continuous_update = False)

    # actual widgets channel 3
    channel_check_c3 = widgets.Checkbox(description="Channel 3")
    R3 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'R', continuous_update = False)
    G3 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'G', continuous_update = False)
    B3 = widgets.IntSlider(value=123, min = 0, max = 255, step = 1, description = 'B', continuous_update = False)
#     time_pt_c3 = widgets.IntSlider(value=0, min=0, max=0, step=1, description='Time pt', continuous_update = False)
    color_text_c3 = widgets.Text(value='', description="color name", continuous_update = False)
    color_hex_c3 = widgets.Text(value='', description="hex color",continuous_update = False)
    file_name_c3 = widgets.Dropdown(options = file_list, description='File')
    inten_min_c3 = widgets.IntSlider(value = img_min, min = img_min, max = img_max, step = 1, description = 'Min Inten', continuous_update = False)
    inten_max_c3 = widgets.IntSlider(value = img_max, min = img_min, max = img_max, step = 1, description = 'Max Inten', continuous_update = False)

    
    # function comes after the sliders because it calls on variables established in the sliders
    def read_image_c1(file_name):
        if type(file_name) == str:
            if file_name[-3:] == 'tif':
                img = io.imread(file_name)
            elif file_name[-3:] == 'czi':
                img = czifile.imread(file_name)
                img = img[0,0,0,:,0,:,:,0]
            else:
                raise Exception("Not a valid string")
                
        if len(img.shape) > 2:
            time_pt.max = img.shape[0] - 1

        img_min = np.min(img)
        img_max = np.max(img)

        inten_min_c1.value = img_min
        inten_max_c1.value = img_max
        inten_min_c1.min = img_min
        inten_min_c1.max = img_max
        inten_max_c1.min = img_min
        inten_max_c1.max = img_max

        return img, img_min, img_max
    
    # function comes after the sliders because it calls on variables established in the sliders
    def read_image_c2(file_name):
        if type(file_name) == str:
            if file_name[-3:] == 'tif':
                img = io.imread(file_name)
            elif file_name[-3:] == 'czi':
                img = czifile.imread(file_name)
                img = img[0,0,1,:,0,:,:,0]
            else:
                raise Exception("Not a valid string")
        if len(img.shape) > 2:
            time_pt.max = img.shape[0] - 1

        img_min = np.min(img)
        img_max = np.max(img)

        inten_min_c2.value = img_min
        inten_max_c2.value = img_max
        inten_min_c2.min = img_min
        inten_min_c2.max = img_max
        inten_max_c2.min = img_min
        inten_max_c2.max = img_max

        return img, img_min, img_max
    
    def read_image_c3(file_name):
        if type(file_name) == str:
            if file_name[-3:] == 'tif':
                img = io.imread(file_name)
            elif file_name[-3:] == 'czi':
                img = czifile.imread(file_name)
                img = img[0,0,2,:,0,:,:,0]
            else:
                raise Exception("Not a valid string")
        if len(img.shape) > 2:
            time_pt.max = img.shape[0] - 1

        img_min = np.min(img)
        img_max = np.max(img)

        inten_min_c3.value = img_min
        inten_max_c3.value = img_max
        inten_min_c3.min = img_min
        inten_min_c3.max = img_max
        inten_max_c3.min = img_min
        inten_max_c3.max = img_max

        return img, img_min, img_max
    
    # update color name based on rgb values
    R1.observe(color_name_fromRGB_c1, names="value")
    G1.observe(color_name_fromRGB_c1, names="value")
    B1.observe(color_name_fromRGB_c1, names="value")

    # update color name based on rgb values
    R2.observe(color_name_fromRGB_c2, names="value")
    G2.observe(color_name_fromRGB_c2, names="value")
    B2.observe(color_name_fromRGB_c2, names="value")
    
    # update color name based on rgb values
    R3.observe(color_name_fromRGB_c3, names="value")
    G3.observe(color_name_fromRGB_c3, names="value")
    B3.observe(color_name_fromRGB_c3, names="value")
    
    # update sliders based on color name or hex value
    color_mapper_c1 = widgets.interactive_output(color_name2RGB_c1, {'color_name' : color_text_c1})
    color_mapperhex_c1 = widgets.interactive_output(hex_to_rgb_c1, {'hex_string' : color_hex_c1})

    # update sliders based on color name or hex value
    color_mapper_c2 = widgets.interactive_output(color_name2RGB_c2, {'color_name' : color_text_c2})
    color_mapperhex_c2 = widgets.interactive_output(hex_to_rgb_c2, {'hex_string' : color_hex_c2})

    # update sliders based on color name or hex value
    color_mapper_c3 = widgets.interactive_output(color_name2RGB_c3, {'color_name' : color_text_c3})
    color_mapperhex_c3 = widgets.interactive_output(hex_to_rgb_c3, {'hex_string' : color_hex_c3})
    
    # update min/max sliders based on choosing a new image in the list
    file_chooser_c1 = widgets.interactive_output(read_image_c1, {'file_name' : file_name_c1})

    # update min/max sliders based on choosing a new image in the list
    file_chooser_c2 = widgets.interactive_output(read_image_c2, {'file_name' : file_name_c2})
    
    # update min/max sliders based on choosing a new image in the list
    file_chooser_c3 = widgets.interactive_output(read_image_c3, {'file_name' : file_name_c3})


    # this is the actual widget to display the proper image based on the sliders/textboxes/etc
    channel1 = widgets.interactive_output(interactive_plot, 
                                    {'channel_check' : channel_check_c1, 'R' : R1, 'G' : G1, 'B' : B1, 'img' : file_name_c1, 't' : time_pt,
                                    'intensity_minimum' : inten_min_c1, 'intensity_maximum' : inten_max_c1, 'c' : fixed(0),
                                    'inverted_check' : inverted_check})

    # this is the actual widget to display the proper image based on the sliders/textboxes/etc
    channel2 = widgets.interactive_output(interactive_plot, 
                                    {'channel_check' : channel_check_c2, 'R' : R2, 'G' : G2, 'B' : B2, 'img' : file_name_c2, 't' : time_pt,
                                    'intensity_minimum' : inten_min_c2, 'intensity_maximum' : inten_max_c2, 'c' : fixed(1),
                                    'inverted_check' : inverted_check})
    
    # this is the actual widget to display the proper image based on the sliders/textboxes/etc
    channel3 = widgets.interactive_output(interactive_plot, 
                                    {'channel_check' : channel_check_c3, 'R' : R3, 'G' : G3, 'B' : B3, 'img' : file_name_c3, 't' : time_pt,
                                    'intensity_minimum' : inten_min_c3, 'intensity_maximum' : inten_max_c3, 'c' : fixed(2),
                                    'inverted_check' : inverted_check})
    # overlay channel
    channel_overlay = widgets.interactive_output(overlay_interactive, 
                                {'inverted_check' : inverted_check, 't' : time_pt,
                                 'img1_check' : channel_check_c1, 'img2_check' : channel_check_c2, 'img3_check' : channel_check_c3,
                                 'img1' : file_name_c1, 'img2' : file_name_c2, 'img3' : file_name_c3,
                                 'R1' : R1, 'G1' : G1, 'B1' : B1,
                                 'R2' : R2, 'G2' : G2, 'B2' : B2,
                                 'R3' : R3, 'G3' : G3, 'B3' : B3,
                                'img1_min' : inten_min_c1, 'img2_min' : inten_min_c2, 'img3_min' : inten_min_c3,
                                'img1_max' : inten_max_c1, 'img2_max' : inten_max_c2, 'img3_max' : inten_max_c3,})

    def save_image(yup):
        channel_list, cmap_list, icmap_list, img_min_list, img_max_list = [], [], [], [], []
        if channel_check_c1.value:
            channel_list.append(file_name_c1.value)
            cmp1, icmp1 = make_colormap((R1.value,G1.value,B1.value),'COLOR',False)
            cmap_list.append(cmp1)
            icmap_list.append(icmp1)
            img_min_list.append(inten_min_c1.value)
            img_max_list.append(inten_max_c1.value)
        if channel_check_c2.value:
            channel_list.append(file_name_c2.value)
            cmp2, icmp2 = make_colormap((R2.value,G2.value,B2.value),'COLOR',False)
            cmap_list.append(cmp2)
            icmap_list.append(icmp2)
            img_min_list.append(inten_min_c2.value)
            img_max_list.append(inten_max_c2.value)
        if channel_check_c3.value:
            channel_list.append(file_name_c3.value)
            cmp3, icmp3 = make_colormap((R3.value,G3.value,B3.value),'COLOR',False)
            cmap_list.append(cmp3)
            icmap_list.append(icmp3)
            img_min_list.append(inten_min_c3.value)
            img_max_list.append(inten_max_c3.value)
        if inverted_check.value:
            overlay_im = overlay_image_widget(channel_list,icmap_list,img_min_list,img_max_list)
#             image_list = image_min_max(file_list, icmap_list, img_min_list, img_max_list)
        else:
#             image_list = image_min_max(file_list, cmap_list, img_min_list, img_max_list)
            overlay_im = overlay_image_widget(channel_list,cmap_list,img_min_list,img_max_list)
        curr_path = file_name_c1.value[:file_name_c1.value.rfind('/')+1]
        io.imsave(curr_path + save_name.value + '.tif' ,overlay_im)

    # save_button.on_click(save_image, file_name.value,R.value,G.value,B.value,inten_min.value,inten_max.value)
    save_button.on_click(save_image)
    
    # display the actual widget
#     display(widgets.VBox([channel_check_c1,file_name_c1,time_pt_c1,color_text_c1,color_hex_c1,R1,G1,B1,inten_min_c1,inten_max_c1,inverted_check,channel1,save_button]))
    display(
        widgets.HBox([
            widgets.VBox([channel_check_c1,file_name_c1,time_pt,color_text_c1,color_hex_c1,R1,G1,B1,
                      inten_min_c1,inten_max_c1,channel1]),
            widgets.VBox([channel_check_c2,file_name_c2,time_pt,color_text_c2,color_hex_c2,R2,G2,B2,
                      inten_min_c2,inten_max_c2,channel2]),
            widgets.VBox([channel_check_c3,file_name_c3,time_pt,color_text_c3,color_hex_c3,R3,G3,B3,
                      inten_min_c3,inten_max_c3,channel3]),
            widgets.VBox([blank,blank,blank,inverted_check, blank,save_name, blank, save_button,blank,blank,channel_overlay])
                    ])
            )


def movie_maker_widget(timelapse_path):
    overlay_timelapse = io.imread(timelapse_path)
    N_frames, img_h, img_w, N_notused = overlay_timelapse.shape

    def update_scaleposition(change):
        if scaleposition_r.value == 20:
            if scaleposition_c.value == 20:
                scaleposition.value = 'top_left'
            elif scaleposition_c.value == int(img_w - 20 - scalebar_length.value/um_per_pix.value):
                scaleposition.value = 'top_right'
            else:
                scaleposition.value = ''
        elif scaleposition_r.value == (img_h - 20):
            if scaleposition_c.value == int(img_w - 20 - scalebar_length.value/um_per_pix.value):
                scaleposition.value = 'bottom_right'
            elif scaleposition_c.value == 20:
                scaleposition.value = 'bottom_left'
            else:
                scaleposition.value = ''
        else:
            scaleposition.value = ''

    def update_scaleposition_rc(change):
        if scaleposition.value == 'bottom_right':
            scaleposition_r.value = img_h - scaleheight.value - 20
            scaleposition_c.value = int(img_w - 20 - scalebar_length.value/um_per_pix.value)
            scalefontposition_r.value = int(scaleposition_r.value - scalefontsize.value - 5)
            scalefontposition_c.value = scaleposition_c.value
        if scaleposition.value == 'bottom_left':
            scaleposition_r.value = img_h - scaleheight.value - 20
            scaleposition_c.value = 20
            scalefontposition_r.value = int(scaleposition_r.value - scalefontsize.value - 5)
            scalefontposition_c.value = scaleposition_c.value
        if scaleposition.value == 'top_left':
            scaleposition_r.value = 20
            scaleposition_c.value = 20
            scalefontposition_r.value = int(scaleposition_r.value + scaleheight.value + 5)
            scalefontposition_c.value = 20
        if scaleposition.value == 'top_right':
            scaleposition_r.value = 20
            scaleposition_c.value = int(img_w - 20 - scalebar_length.value/um_per_pix.value)
            scalefontposition_r.value = int(scaleposition_r.value + scaleheight.value + 5)
            scalefontposition_c.value = scaleposition_c.value

    def update_timeposition(change):
        if timeposition_r.value == 30:
            if timeposition_c.value == 30:
                timeposition.value = 'top_left'
            elif timeposition_c.value == (img_w - 3 * timefontsize.value - 30):
                timeposition.value = 'top_right'
            else:
                timeposition.value = ''
        elif timeposition_r.value == (img_h - 1 * timefontsize.value):
            if timeposition_c.value == (img_w - 3 * timefontsize.value - 30):
                timeposition.value = 'bottom_right'
            elif timeposition_c.value == 30:
                timeposition.value = 'bottom_left'
            else:
                timeposition.value = ''
        else:
            timeposition.value = ''

    def update_timeposition_rc(change):
        if change.new == 'bottom_right':
            timeposition_r.value = img_h - 1 * timefontsize.value
            timeposition_c.value = img_w - 3 * timefontsize.value - 30
        if change.new == 'bottom_left':
            timeposition_r.value = img_h - 1 * timefontsize.value
            timeposition_c.value = 30
        if change.new == 'top_left':
            timeposition_r.value = 30
            timeposition_c.value = 30
        if change.new == 'top_right':
            timeposition_r.value = 30
            timeposition_c.value = img_w - 3 * timefontsize.value - 30

    def update_scalebar_length(change):
        label_str = scalelabel.value
        if len(label_str) > 0:
            numbers = []
            for word in label_str.split():
                if word.isdigit():
                      numbers.append(int(word))
            scalebar_length.value = int(numbers[0])


    def save_image(yup):
        curr_path = timelapse_path[:timelapse_path.rfind('/')+1]
        savename = curr_path + filesavename.value + '.mp4'
        # write out the parameters as a csv file for reference later
        if save_params_check.value:
            param_dict = {
                'time_interval' : time_interval.value,
                'um_per_pix' : um_per_pix.value,
                'scalebar_check' : scalebar_check.value,
                'scalebar_length' : scalebar_length.value,
                'scaleheight' : scaleheight.value,
                'scalecolor' : scalecolor.value,
                'scaleposition_r' : scaleposition_r.value,
                'scaleposition_c' : scaleposition_c.value,
                'scalelabel' : scalelabel.value,
                'scalefontsize' : scalefontsize.value,
                'scalefont_r' : scalefontposition_r.value,
                'scalefont_c' : scalefontposition_c.value,
                'timestamp_check' : timestamp_check.value,
                'timeposition_r' : timeposition_r.value,
                'timeposition_c' : timeposition_c.value,
                'timefontsize' : timefontsize.value,
                'framerate' : framerate.value,
                'quality' : quality.value
            }
            with open(curr_path + filesavename.value + '_params.csv', 'w', newline="") as csv_file:  
                writer = csv.writer(csv_file)
                for key, value in param_dict.items():
                    writer.writerow([key, value])

        if timestamp_check.value:
            # figure out the format
            if (time_interval.value * N_frames) < 60:
                fmt = 'sec:'
            elif (time_interval.value * N_frames) < 3600:
                fmt = 'min:sec'
            else:
                fmt = 'hr:min:sec'
            # make the timestamps
            time_pts = make_timestamp_list(time_interval.value, N_frames, fmt)

            if scalebar_check.value:
                save_timelapse_as_movie(savename, overlay_timelapse, 
                                        scalebar_length = int(scalebar_length.value / um_per_pix.value),
                                        scaleheight = scaleheight.value, scalecolor = scalecolor.value, 
                                        scaleposition = (scaleposition_r.value, scaleposition_c.value), 
                                        scalelabel = scalelabel.value, scalefontsize = scalefontsize.value, 
                                        scalefontposition = (scalefontposition_r.value, scalefontposition_c.value),
                                        time_pts = time_pts, timeposition = (timeposition_r.value, timeposition_c.value), 
                                        timefontsize = timefontsize.value, framerate = framerate.value, quality = quality.value)
            else:
                save_timelapse_as_movie(savename, overlay_timelapse, 
                                        scalecolor = scalecolor.value,
                                        time_pts = time_pts, timeposition = (timeposition_r.value, timeposition_c.value), 
                                        timefontsize = timefontsize.value, framerate = framerate.value, quality = quality.value)

        elif scalebar_check.value:
            save_timelapse_as_movie(savename, overlay_timelapse, 
                                    scalebar_length = int(scalebar_length.value / um_per_pix.value),
                                    scaleheight = scaleheight.value, scalecolor = scalecolor.value, 
                                    scaleposition = (scaleposition_r.value, scaleposition_c.value), 
                                    scalelabel = scalelabel.value, scalefontsize = scalefontsize.value, 
                                    scalefontposition = (scalefontposition_r.value, scalefontposition_c.value),
                                    framerate = framerate.value, quality = quality.value)
        else:
            save_timelapse_as_movie(savename, overlay_timelapse, 
                                    framerate = framerate.value, quality = quality.value)

    def preview_overlay(imstack, movie_frame, time_interval, um_per_pix, scalebar_check, scalebar_length, scaleheight, scalecolor, scaleposition_r, scaleposition_c, 
                        scalelabel, scalefontsize, scalefont_r, scalefont_c, timestamp_check, timeposition_r, timeposition_c, timefontsize, framerate, 
                        quality, N_frames):
        scaleposition = (scaleposition_r,scaleposition_c)
        scalefontposition = (scalefont_r, scalefont_c)
        timeposition = (timeposition_r,timeposition_c)
        scalebar_length_um = int(scalebar_length / um_per_pix)

        # check if a timestamp is included
        if timestamp_check:
            # figure out the format
            if (time_interval * N_frames) < 60:
                fmt = 'sec:'
            elif (time_interval * N_frames) < 3600:
                fmt = 'min:sec'
            else:
                fmt = 'hr:min:sec'
            # make the timestamps
            time_pts = make_timestamp_list(time_interval, N_frames, fmt)

            # if so save with metadata tag
            io.imsave('first_frame.tif',imstack[movie_frame], plugin='tifffile',
                        extratags=[(306, 's', 0, str(time_pts[movie_frame]), True )])
        else:
            # just save normally
            io.imsave('first_frame.tif',imstack[movie_frame])



        # generate the ffmpeg command
        if scalebar_check:
            if timestamp_check:
                command_string, params = ffmpeg_str('first_frame_overlay.tif', imstack, fmt = 't', 
                                                    scalebar_length=scalebar_length_um, scaleheight=scaleheight,
                                                    scalecolor=scalecolor, scaleposition=scaleposition,
                                                    scalelabel=scalelabel, scalefontsize=scalefontsize, 
                                                    scalefontposition = scalefontposition,
                                                    time_pts = time_pts, timeposition=timeposition,
                                                    timefontsize=timefontsize, framerate=framerate, quality=quality)
            else:
                command_string, params = ffmpeg_str('first_frame_overlay.tif', imstack, fmt = 't', 
                                                    scalebar_length=scalebar_length_um, scaleheight=scaleheight,
                                                    scalecolor=scalecolor, scaleposition=scaleposition,
                                                    scalelabel=scalelabel, scalefontsize=scalefontsize,
                                                    scalefontposition = scalefontposition,
                                                    framerate=framerate, quality=quality)
        elif timestamp_check:
            command_string, params = ffmpeg_str('first_frame_overlay.tif', imstack, fmt = 't', 
                                                time_pts = time_pts, timeposition=timeposition,timecolor=scalecolor,
                                                timefontsize=timefontsize, framerate=framerate, quality=quality)
        else:
            command_string, params = ffmpeg_str('first_frame_overlay.tif', imstack, fmt = 't', 
                                                framerate=framerate, quality=quality)

        # run ffmpeg
        subprocess.call(command_string, shell=True)

        # delete the first frame
        os.remove('first_frame.tif')

        # read in the overlay
        first_frame_overlay = io.imread('first_frame_overlay.tif')

        # plot the overlay
        fig = plt.figure(figsize=(8,8))
        plt.imshow(first_frame_overlay)
        plt.axis('off')



    blank = widgets.Label(value = '')
    input_details = widgets.Label(value = 'Timelapse Movie Details')
    time_interval = widgets.FloatText(value = 1, description = 'timestep (s)', continuous_update = False)
    um_per_pix = widgets.FloatText(value = .17460, description='µm/pixel', continuous_update = False)
    movie_frame = widgets.IntSlider(value=0, min=0, max=N_frames-1, description='Frame', continuous_update = False)
    save_params_check = widgets.Checkbox(description = 'Save Parameters Text File')

    scalebar_details = widgets.Label(value = 'Scalebar Parameters')
    scalebar_check = widgets.Checkbox(description = 'Include Scalebar')
    scalebar_length = widgets.IntSlider(value=10, min=0, max= int(img_w * 0.5 * um_per_pix.value), description='length (µm)', continuous_update=False)
    scaleheight = widgets.IntSlider(value=int(img_h * 0.015), min = 1, max = 100, step = 1, description = 'scaleheight', continuous_update = False)
    scaleposition = widgets.Dropdown(options=['','bottom_right','bottom_left','top_left','top_right'],value='bottom_right', description='scale position')
    scaleposition_r = widgets.IntSlider(value=(img_h - 20), min=0, max=img_h, description='scalebar row', continuous_update=False)
    scaleposition_c = widgets.IntSlider(value=int(img_w - 20 - scalebar_length.value/um_per_pix.value), min=0, max=img_w, description='scalebar col', continuous_update=False)
    scalelabel = widgets.Text(value='10 µm', placeholder = '10 µm', description = 'scale label', continuous_update = False)
    scalefontsize = widgets.IntSlider(value=(int(scalebar_length.value / um_per_pix.value / 3)), min = 6, max=130, step = 1, description = 'fontsize', continuous_update = False)
    scalefontposition_r = widgets.IntSlider(value=(scaleposition_r.value - scalefontsize.value - 5), min=0, max=img_h, description='font row', continuous_update=False)
    scalefontposition_c = widgets.IntSlider(value=int(img_w - 20 - scalebar_length.value/um_per_pix.value), min=0, max=img_w, description='font col', continuous_update=False)

    timestamp_details = widgets.Label(value = 'Time Stamp Parameters')
    timestamp_check = widgets.Checkbox(description = 'Include Time Stamp')
    timefontsize = widgets.IntSlider(value=(3 * scalefontsize.value), min = 6, max=130, step = 1, description = 'fontsize', continuous_update = False)
    timeposition = widgets.Dropdown(options=['','bottom_right','bottom_left','top_left','top_right'], value='top_right', description='time position')
    timeposition_r = widgets.IntSlider(value=30, min=0, max=img_h, description='time row', continuous_update=False)
    timeposition_c = widgets.IntSlider(value=(img_w - 3 * timefontsize.value - 30), min=0, max=img_w, description='time col', continuous_update=False)
    
    movie_details = widgets.Label(value = 'Output Movie Parameters')
    quality = widgets.IntSlider(value=25, min = 15, max = 30, step = 1, description = 'quality', continuous_update = False)
    framerate = widgets.IntSlider(value=10, min = 1, max = 60, step = 1, description = 'framerate', continuous_update = False)
    scalecolor = widgets.Dropdown(options=['black', 'white', 'gray'], description='Text color')
    filesavename = widgets.Text(value='', description='Save Name', continuous_update = False)
    savebutton = widgets.Button(description="Save Movie", layout=Layout(width='98%'))
    preview_overlay_button = widgets.Button(description="Preview Overlay", layout=Layout(width='98%'))


    scaleposition_r.observe(update_scaleposition, names="value")
    scaleposition_c.observe(update_scaleposition, names="value")
    timeposition_r.observe(update_timeposition, names="value")
    timeposition_c.observe(update_timeposition, names="value")

    scaleposition.observe(update_scaleposition_rc, names="value")
    timeposition.observe(update_timeposition_rc, names="value")

    scalebar_length.observe(update_scaleposition_rc, names="value")

    um_per_pix.observe(update_scalebar_length, names="value")
    scalelabel.observe(update_scalebar_length, names="value")

    savebutton.on_click(save_image)
    preview_overlay_button.on_click(preview_overlay)

    # overlay channel
    overlay_preview = widgets.interactive_output(preview_overlay, 
                            {'imstack' : fixed(overlay_timelapse), 'movie_frame' : movie_frame, 'time_interval' : time_interval, 
                             'um_per_pix' : um_per_pix,
                            'scalebar_check' : scalebar_check, 'scalebar_length'  : scalebar_length, 'scaleheight' : scaleheight,
                            'scalecolor' : scalecolor, 'scaleposition_r' : scaleposition_r, 'scaleposition_c' : scaleposition_c, 
                             'scalelabel' : scalelabel, 'scalefontsize' : scalefontsize, 'scalefont_r': scalefontposition_r,
                             'scalefont_c' : scalefontposition_c, 'timestamp_check' : timestamp_check, 
                             'timeposition_r' : timeposition_r, 'timeposition_c' : timeposition_c, 'timefontsize' : timefontsize, 
                             'framerate' : framerate, 'quality' : quality, 'N_frames' : fixed(N_frames)})


    # preview_overlay(imstack, scalebar_check, scalebar_length, scaleheight, scalecolor, scaleposition, scalelabel, scalefontsize,
    #                    timestamp_check, timeposition, timefontsize, framerate, quality, N_frames)

    display(widgets.HBox([widgets.VBox([input_details,time_interval,um_per_pix, movie_frame,
                          blank, scalebar_details, scalebar_check, scaleposition, scaleposition_r, 
                          scaleposition_c,scalebar_length, scaleheight,scalelabel,scalefontsize, 
                          scalefontposition_r, scalefontposition_c], layout=Layout(display='flex' ,
    align_items='center')),
            widgets.VBox([movie_details, quality, framerate, scalecolor,blank,
                          timestamp_details, timestamp_check, timeposition, timeposition_r, 
                          timeposition_c, timefontsize,blank,blank,filesavename,save_params_check,savebutton], 
                         layout=Layout(display='flex' , align_items='center')),
            widgets.VBox([overlay_preview])
                         ])
           )

