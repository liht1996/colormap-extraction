import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

%matplotlib inline 

def find_most_frequent_color(img):
    data = np.reshape(img, (-1,3))
    dict_count = {}

    for i in range(data.shape[0]):
        if str(data[i,:].tolist()) not in dict_count.keys():
            dict_count[str(data[i,:].tolist())] = 1
        else:
            dict_count[str(data[i,:].tolist())] += 1
            
    color_str = sorted(dict_count.items(), key=lambda x:x[1], reverse=1)[0][0]
    return np.fromstring(color_str[1:-1], dtype=int, sep=',')

def mask_img(img, mfc, T_bg=5):
    
    mask = np.ones((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        # mask for background
            if np.max(abs(img[i,j]-mfc)) <= T_bg:
                img[i,j] = [255,128,128]
                mask[i,j] = 0    
    
    return img, mask

def binarize(img):
    img_gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY).reshape(-1,)
    mask_above_127 = (img_gray > 127)
    mask_below_127 = (img_gray <= 127)
    img_gray[mask_above_127] = 0
    img_gray[mask_below_127] = 1
    
    return img_gray.reshape((img.shape[0], img.shape[1]))

def flood_fill(img, seed_point=(20, 100)):
    return cv2.floodFill(img, None, seed_point, 1)[1]

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_LAB2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    plt.imshow(labeled_img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    
def find_largest_connected_component(img):
    num_labels, labels = cv2.connectedComponents(img)
    dict_area = {}
    for i in range(num_labels):
        dict_area[i] = area_i = np.sum(labels == i)
                  
    return sorted(dict_area.items(), key=lambda x:x[1], reverse=1)[1][0], labels


def get_colors(img, idx, labels):
    positions = np.argwhere(labels==idx)
    position_lefttop = positions[0,:]
    position_rightbottom = positions[-1,:]
    
    # width <= height
    if (position_lefttop[0]-position_rightbottom[0]) <= (position_lefttop[1]-position_rightbottom[1]):
        # most_top_color, most_bottom_color
        return [img[position_lefttop[0], int((position_lefttop[1]+position_rightbottom[1])/2)], \
                img[position_rightbottom[0], int((position_lefttop[1]+position_rightbottom[1])/2)]]
    else:
        return [img[int((position_lefttop[0]+position_rightbottom[0])/2), position_lefttop[1]], \
                img[int((position_lefttop[1]+position_rightbottom[1])/2), position_lefttop[0]]]
    
def get_area_plot_and_save(img, idx, labels, output_path, chart_name):
    positions = np.argwhere(labels==idx)
    position_lefttop = positions[0,:]
    position_rightbottom = positions[-1,:]
    
    area = img[position_lefttop[0]:position_rightbottom[0], position_lefttop[1]:position_rightbottom[1], :]
    
    try:
        cv2.imwrite(output_path+chart_name,cv2.cvtColor(area, cv2.COLOR_LAB2BGR))
    except:
        return
    
    
def get_result(charts_path, output_path, chart_name, seed_point=None):

    previous_pos_lt = []
    previous_pos_rb = []
    
    dict_pos_lt = {}
    dict_pos_rb = {}

    if seed_point:
        k = seed_point[0]
        p = seed_point[1]
        img_bgr = cv2.imread(charts_path+chart_name, cv2.IMREAD_COLOR) 
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        mfc = find_most_frequent_color(img_lab)
        masked_img_lab,mask = mask_img(img_lab, mfc, T_bg=0)
        img_binary = binarize(masked_img_lab)
        img_filled = flood_fill(img_binary, seed_point=(int(img_lab.shape[1]*(k/100.0)), int(img_lab.shape[0]*(p/100.0))))
        img_eroded = erode(img_filled, kernel_size=5)
        idx, labels = find_largest_connected_component(img_eroded)
        n = chart_name.split(".")[0]+f"_{k}_{p}.png"
        get_area_plot_and_save(img_lab, idx, labels, output_path, n)
        dict_pos_lt[f"{k}_{p}"] = list(position_lefttop)
        dict_pos_rb[f"{k}_{p}"] = list(position_rightbottom)
        
    else:
        # brute force method to search seed point for flood fill
        for k in range(10, 100, 10):
            for p in range(10, 100, 10):
                try:
                    img_bgr = cv2.imread(charts_path+chart_name, cv2.IMREAD_COLOR) 
                    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    mfc = find_most_frequent_color(img_lab)
                    masked_img_lab,mask = mask_img(img_lab, mfc, T_bg=0)
                    img_binary = binarize(masked_img_lab)
                    img_filled = flood_fill(img_binary, seed_point=(int(img_lab.shape[1]*(k/100.0)), int(img_lab.shape[0]*(p/100.0))))
                    img_eroded = erode(img_filled, kernel_size=5)
                    idx, labels = find_largest_connected_component(img_eroded)
                    n = chart_name.split(".")[0]+f"_{k}_{p}.png"
                    positions = np.argwhere(labels==idx)
                    position_lefttop = positions[0,:]
                    position_rightbottom = positions[-1,:]

                    if not list(position_lefttop) in previous_pos_lt or\
                    not list(position_rightbottom) in previous_pos_rb:
                        previous_pos_lt.append(list(position_lefttop))
                        previous_pos_rb.append(list(position_rightbottom))
                        get_area_plot_and_save(img_lab, idx, labels, output_path, n)
                        dict_pos_lt[f"{k}_{p}"] = list(position_lefttop)
                        dict_pos_rb[f"{k}_{p}"] = list(position_rightbottom)
                except:
                    print(chart_name, k, p)
                
    return {
        "lt": dict_pos_lt,
        "rb": dict_pos_rb
    }