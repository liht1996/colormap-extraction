import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
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

def mask_img(img, mfc, T_bg=5, T_gray=5):
    
#     mask = np.ones((img.shape[0], img.shape[1]), dtype='uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
        # mask for background
            if np.max(abs(img[i,j,:]-mfc)) <= T_bg:
                img[i,j] = [255,128,128]
            # mask for gray
            elif np.max(abs(img[i,j,1:]-np.array([128, 128]))) <= T_gray:
                img[i,j] = [255,128,128]
    
    return img

def color_clustering(img, eps=5, min_samples=20):
    data = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            data.append(np.concatenate((img[i,j,:], np.array([i,j]))))
            
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(data)) 
    
    color_dict = {}
    
    # find colors of pixels in each cluster
    for p in range(max(clustering.labels_)+1):
        color_dict[p] = img.reshape(-1,3)[clustering.labels_==p]
        
    # find most frequent color in each cluster
    list_color = []
    for q in color_dict.values():
        mfc_ = find_most_frequent_color(q)
        # remove masked colors
        if mfc_.tolist() != [255, 128, 128]:
            list_color.append(mfc_)
    
    return list_color

def plot_color(colors):
    color_matrices = []
    for i in colors:
        color_matrices.append(np.full((20, 20, 3), i.tolist(), dtype='uint8'))
        
    img = np.concatenate(color_matrices, axis=1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_LAB2RGB))
    plt.axis('off')
    plt.show()   
    
def save_color(colors, output_path, chart_name):
    color_matrices = []
    for i in colors:
        color_matrices.append(np.full((80, 80, 3), i.tolist(), dtype='uint8'))
        
    img = np.concatenate(color_matrices, axis=1)
    cv2.imwrite(output_path+chart_name, cv2.cvtColor(img, cv2.COLOR_LAB2BGR))
    
def get_result(charts_path, output_path, chart_name):
    img_bgr = cv2.imread(charts_path+chart_name, cv2.IMREAD_COLOR) 
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    mfc = find_most_frequent_color(img_lab)
    masked_img_lab = mask_img(img_lab, mfc)
    colors = color_clustering(masked_img_lab, eps=5, min_samples=10)
    
    save_color(colors, output_path, chart_name)
    
    return [list(i) for i in colors]