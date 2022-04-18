# -*- coding: utf-8 -*-

import os
import numpy as np
# for reading and processing images
import cv2
import tensorflow as tf
import config as cfu
from tqdm import tqdm


# Create dir
def create_dir(path):
    if not os.path.exists(path):
        print("Hi from create dir")
        os.mkdir(path)

# Avoid OOM errors by setting GPU Memory Consumpthion
def set_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
# Sort image and label file name         
def load_data(path_img, path_label): 
    """
    Looks for relevant filenames in the shared path 
    Return: two list for original and labeled files respectively
    """
    
    # Read the images folder like a list
    img_list = os.listdir(path_img)
    labels_list = os.listdir(path_label)

    sorted_img_arr = []
    sorted_label_arr = []
    
    for file in img_list:
        sorted_img_arr.append(file)
    for file in labels_list:
        sorted_label_arr.append(file)
    
    sorted_img_arr.sort()
    sorted_label_arr.sort()
    
    return sorted_img_arr, sorted_label_arr


# Read and resize original image
def get_imageArr(path, width, height):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    return img

# Read and resize seg image
def get_segmentationArr(path, width, height):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    img = img[: , : , 0]
    return img

# Array of all images 
#  
def all_image_array(org_img_dir, seg_img_dir, list_mapped, width, height):
    X = []
    Y = []

    for i in tqdm(range(len(list_mapped))):
        
        X.append(get_imageArr(os.path.join(org_img_dir, list_mapped[i][0]), width, height))
        Y.append(get_segmentationArr(os.path.join(seg_img_dir, list_mapped[i][1]), width, height))

    return np.array(X), np.array(Y)



def augment_data(train_img_dir, train_seg_dir, list_mapped):
    X = []
    Y = []
    
    for i in tqdm(range(len(list_mapped))):
        
        image = cv2.imread(os.path.join(train_img_dir, list_mapped[i][0]), 1)
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=(1, 2))
        image = tf.image.stateless_random_contrast(image, lower=0.2, upper=0.5, seed=(1, 2))
        image = tf.image.stateless_random_jpeg_quality(image, 75, 95, (1, 2))
        image = tf.image.stateless_random_saturation(image, 0.5, 1.0, (1, 2))
        X.append(image)
        
        label = cv2.imread(os.path.join(train_seg_dir, list_mapped[i][1]), 1)
        label = label[:, :, 0]
        Y.append(label)
        
    return np.array(X), np.array(Y)

# Sperate part of train image as calib image and store the image in directory
def store_train_cal_img(X_train, Y_train, t_img_dir, t_seg_dir, c_img_dir, c_seg_dir):
    trn_count = int(0)
    cal_cnt = int(0)

    for i in tqdm(range(len(X_train))):
        img = X_train[i]
        seg = Y_train[i]
        
        cv2.imwrite(t_img_dir + '/training_' + str(trn_count) + '.png', img)
        cv2.imwrite(t_seg_dir + '/seg_trn_'+ str(trn_count) + '.png', seg)
        
        if ((trn_count%int(8)) == int(0)):
            cv2.imwrite(c_img_dir +  '/calib_'+ str(trn_count) + '.png', img)
            cv2.imwrite(c_seg_dir + '/seg_clb_'+ str(trn_count) + '.png', seg)
            cal_cnt += 1
            
        trn_count += 1

def store_img(X, Y, img_dir, seg_dir, img_name, seg_name):
    count = int(0)
    for i in tqdm(range(len(X))):
        img = X[i]
        seg = Y[i]
        
        cv2.imwrite(img_dir + img_name + str(count) + '.png', img)
        cv2.imwrite(seg_dir + seg_name + str(count) + '.png', seg)
        
        count += 1

def store_aug_img(X, Y, img_dir, seg_dir, img_name, seg_name):
    count = len(X)
    for i in tqdm(range(len(X))):
        img = X[i]
        seg = Y[i]
        
        cv2.imwrite(img_dir + img_name + str(count) + '.png', img)
        cv2.imwrite(seg_dir + seg_name + str(count) + '.png', seg)
        
        count += 1

# Preprocessing data
def preprocess_data(images, masks, target_shape_img, target_shape_mask, path_img, path_mask):
    
    # Pull the relevant dimentions of the image and mask 
    total_img = len(images)
    
    h_img, w_img, c_img = target_shape_img
    h_mask, w_mask, c_mask = target_shape_mask
    
    # Define X and y with number of images along with the shape of one image
    X = np.zeros((total_img, h_img, w_img, c_img), dtype = np.float32)
    y = np.zeros((total_img, h_mask, w_mask, c_mask), dtype = np.int32)
    
    # Resize images and mask
    for file in tqdm(images):
        index = images.index(file)
        path = os.path.join(path_img, file)
        image = cv2.imread(path, 1)
        image = cv2.resize(image, (w_img, h_img))
        image = image/255
        X[index] = image
        
#     for file in mask: 
        # Concert image into an array of desired shape
        mask_index = masks[index]
        path = os.path.join(path_mask, mask_index)
        single_mask = cv2.imread(path, 1)
        single_mask = cv2.resize(single_mask, (w_mask, h_mask))
        single_mask = single_mask[:, :, 0]
        single_mask = single_mask[ : , : , tf.newaxis] # for convinence to start class from 0 (1, 2, 3) -> (0, 1, 2)
        y[index] = single_mask
    return X, y


def give_color_to_seg_img(seg,n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = cfu.COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return(seg_img)





def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    print("I am in IOU")
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    #Nclass = cfu.NCLASS
    print("Value of Class: ", Nclass)
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi == c))
        FP = np.sum( (Yi != c)&(y_predi == c))
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        #print("class {:02.0f}: #TP={:7.0f}, #FP={:7.0f}, #FN={:7.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
#        print("class (%2d) %12.12s: #TP=%7.0f, #FP=%7.0f, #FN=%7.0f, IoU=%4.3f" % (c, uce.CLASS_NAMES[c],TP,FP,FN,IoU))
        print("class (%2d) %12.12s: #TP=%7.0f, #FP=%7.0f, #FN=%7.0f, IoU=%4.3f" % (c, cfu.CLASS_NAMES[c],TP,FP,FN,IoU))
        print("Value of c: ", c)
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
    return


