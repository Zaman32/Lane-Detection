# -*- coding: utf-8 -*-
import os

WORKING_DIR = os.getcwd()
IMAGES_DIR = os.path.join(WORKING_DIR, 'datasets', 'images')
LABELS_DIR = os.path.join(WORKING_DIR, 'datasets', 'labels')

# =============================================================================
# 
# # Dir paths
# TEST_IMG = os.path.join(WORKING_DIR, 'datasets', 'test_imgs_224')
# TEST_LABEL = os.path.join(WORKING_DIR, 'datasets', 'test_labels_224')
# 
# TRAIN_IMG = os.path.join(WORKING_DIR, 'datasets', 'train_imgs_224')
# TRAIN_LABEL = os.path.join(WORKING_DIR, 'datasets', 'train_labels_224')
# 
# CALIB_IMG = os.path.join(WORKING_DIR, 'datasets', 'calib_imgs_224')
# CALIB_LABEL = os.path.join(WORKING_DIR, 'datasets', 'calib_labels_224')
# =============================================================================

# =============================================================================
# 
# # Dir paths
# TEST_IMG = os.path.join(WORKING_DIR, 'datasets', 'test_imgs_128')
# TEST_LABEL = os.path.join(WORKING_DIR, 'datasets', 'test_labels_128')
# 
# TRAIN_IMG = os.path.join(WORKING_DIR, 'datasets', 'train_imgs_128')
# TRAIN_LABEL = os.path.join(WORKING_DIR, 'datasets', 'train_labels_128')
# 
# CALIB_IMG = os.path.join(WORKING_DIR, 'datasets', 'calib_imgs_128')
# CALIB_LABEL = os.path.join(WORKING_DIR, 'datasets', 'calib_labels_128')
# =============================================================================

"""
# Dir paths
TEST_IMG = os.path.join(WORKING_DIR, 'datasets', 'small', 'test_img')
TEST_LABEL = os.path.join(WORKING_DIR, 'datasets', 'small', 'test_labels')

TRAIN_IMG = os.path.join(WORKING_DIR, 'datasets', 'small', 'train_img')
TRAIN_LABEL = os.path.join(WORKING_DIR, 'datasets', 'small', 'train_labels')

CALIB_IMG = os.path.join(WORKING_DIR, 'datasets', 'small', 'calib_img')
CALIB_LABEL = os.path.join(WORKING_DIR, 'datasets', 'small', 'calib_labels')
"""

# =============================================================================
# # Dir paths
# TEST_IMG = os.path.join(WORKING_DIR, 'datasets', 'large', 'test_img')
# TEST_LABEL = os.path.join(WORKING_DIR, 'datasets', 'large', 'test_labels')
# 
# TRAIN_IMG = os.path.join(WORKING_DIR, 'datasets', 'large', 'train_img')
# TRAIN_LABEL = os.path.join(WORKING_DIR, 'datasets', 'large', 'train_labels')
# 
# CALIB_IMG = os.path.join(WORKING_DIR, 'datasets', 'large', 'calib_img')
# CALIB_LABEL = os.path.join(WORKING_DIR, 'datasets', 'large', 'calib_labels')
# 
# =============================================================================


# Dir paths
TEST_IMG = os.path.join(WORKING_DIR, 'datasets', '224_large', 'test_img')
TEST_LABEL = os.path.join(WORKING_DIR, 'datasets', '224_large', 'test_labels')

TRAIN_IMG = os.path.join(WORKING_DIR, 'datasets', '224_large', 'train_img')
TRAIN_LABEL = os.path.join(WORKING_DIR, 'datasets', '224_large', 'train_labels')

CALIB_IMG = os.path.join(WORKING_DIR, 'datasets', '224_large', 'calib_img')
CALIB_LABEL = os.path.join(WORKING_DIR, 'datasets', '224_large', 'calib_labels')


HEIGHT = 224
WIDTH  = 224
# =============================================================================
# HEIGHT = 128
# WIDTH  = 128
# =============================================================================





NCLASS = 10
LANE_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# colors for segmented classes\n",
COLORS = [[   0,   0, 255],
          [   0, 255,   0],
          [   0, 255, 255],        
          [ 255,   0,   0],
          [ 255,   0, 255],
          [ 255, 255,   0],
          [ 255, 128,  30],
          [   0, 128,  30]]

EPOCHS = 100
BATCH_SIZE = 4


# Name of classes
# =============================================================================
# CLASS_NAMES = ("Single White Solid",
#                "Single White Dotted",
#                "Single Yellow Solid",
#                "Single Yellow Dotted",
#                "Double White Solid",
#                "Double Yellow Solid",
#                "Double Yellow Dotted",
#                "Double White Solid Dotted",
#                "Double White Dotted Solid",
#                "Double Solid White and Yellow")
# 
# 
# =============================================================================
CLASS_NAMES = ("SWS",
               "SWD",
               "SYS",
               "SYD",
               "DWS",
               "DYS",
               "DYD",
               "DWSD",
               "DWDS",
               "DSWY")