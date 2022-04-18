# -*- coding: utf-8 -*-
from random import shuffle, seed

import os
import config as cf
import utils as ut
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Create Dir
ut.create_dir(cf.TRAIN_IMG)
ut.create_dir(cf.TRAIN_LABEL)

ut.create_dir(cf.TEST_IMG)
ut.create_dir(cf.TEST_LABEL)

ut.create_dir(cf.CALIB_IMG)
ut.create_dir(cf.CALIB_LABEL)

image_list, label_list = ut.load_data(cf.IMAGES_DIR, cf.LABELS_DIR)

zipped_list = list(zip(image_list, label_list))

# Take 5 image from each scene
partial_zipped = []
for i in range(len(zipped_list)):
    if i%2 ==0:
        partial_zipped.append(zipped_list[i])
        
print(len(partial_zipped))

seed(1)
shuffle(partial_zipped)


# Seperate train and test set

X, y = ut.all_image_array(cf.IMAGES_DIR, cf.LABELS_DIR, partial_zipped, cf.WIDTH, cf.HEIGHT)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

ut.store_img(X_test, y_test, cf.TEST_IMG, cf.TEST_LABEL, '/test_', '/seg_tst_')

ut.store_train_cal_img(X_train, y_train, cf.TRAIN_IMG, cf.TRAIN_LABEL, cf.CALIB_IMG, cf.CALIB_LABEL)


print(len(os.listdir(cf.TRAIN_IMG)))
print(len(os.listdir(cf.TRAIN_LABEL)))
print(len(os.listdir(cf.TEST_IMG)))
print(len(os.listdir(cf.TEST_LABEL)))
print(len(os.listdir(cf.CALIB_IMG)))
print(len(os.listdir(cf.CALIB_LABEL)))

#print(shuffled_list[:10])
print("----------------------------------------------------------------------")
print("######################### Data Augmentation ###########################")
print("----------------------------------------------------------------------")



train_image_list, train_label_list = ut.load_data(cf.TRAIN_IMG, cf.TRAIN_LABEL)

train_zipped_list = list(zip(train_image_list, train_label_list))

aug_X, aug_y = ut.augment_data(cf.TRAIN_IMG, cf.TRAIN_LABEL, train_zipped_list)
        
ut.store_aug_img(aug_X, aug_y, cf.TRAIN_IMG, cf.TRAIN_LABEL, '/training_', '/seg_trn_')


































