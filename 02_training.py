# -*- coding: utf-8 -*-
print("----------------------------------------------------------------------")
print("###################### Importing requred Modules #####################")
print("----------------------------------------------------------------------")
import config as cf
import utils as ut
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
import my_unet as my_unet
import numpy as np

print("All models have been imported")
###############################################################################




###############################################################################
print("----------------------------------------------------------------------")
print("####################### Set GPU Growth ###############################")
print("----------------------------------------------------------------------")

ut.set_gpu_growth()
print("Setting GPU growth done")
###############################################################################




###############################################################################
print("----------------------------------------------------------------------")
print("####################### Preprocess data ##############################")
print("----------------------------------------------------------------------")

target_img_shape = (cf.HEIGHT, cf.WIDTH, 3)
target_label_shape = (cf.HEIGHT, cf.WIDTH, 1)

img_files, label_files = ut.load_data(cf.TRAIN_IMG, cf.TRAIN_LABEL)

# Take sample of every scene

X, y = ut.preprocess_data(img_files, label_files, target_img_shape, target_label_shape, cf.TRAIN_IMG, cf.TRAIN_LABEL)

plt.imshow(y[0])
print(np.amin(y[0]))
# Shape of images and labels
print(X.shape)
print(y.shape)

fig, arr = plt.subplots(1, 2, figsize=(16, 10))

arr[0].imshow(X[500])
arr[0].set_title("Test Image")
arr[1].imshow(y[500])
arr[1].set_title("Ground Truth")

print("Preprocessing of data has been finished")
###############################################################################





###############################################################################
print("----------------------------------------------------------------------")
print("################# Generate Train, Test and Valid data ################")
print("----------------------------------------------------------------------")           

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size=0.10, 
                                                      random_state=42)  
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_valid shape: ", X_valid.shape)
print("y_valid shape: ", y_valid.shape)

print("Dataset generated")
###############################################################################



###############################################################################
print("----------------------------------------------------------------------")
print("##################### Model Initialization ###########################")
print("----------------------------------------------------------------------") 


###############################################################################







###############################################################################
print("----------------------------------------------------------------------")
print("############################ Training ################################")
print("----------------------------------------------------------------------")

# Callbacks - save the best model
model_save_path = os.path.join(os.getcwd(), 'keras_model', 
                               'my_best_model_epoch{epoch:02d}_acc{accuracy:.2f}.h5')

save_callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_save_path,
    save_best_only=True,
    monitor = 'val_accuracy',
    mode = 'max'
)

early_stop_callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3
)

# Track time

# create model
unet = my_unet.u_net_model(input_shape=(128,128,3), n_filter=32, n_class=10)
unet.summary()



# compile model
unet.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

s_time = datetime.now()

results = unet.fit(X_train, y_train, 
                   batch_size=cf.BATCH_SIZE, 
                   epochs = cf.EPOCHS, 
                   validation_data=(X_valid, y_valid), 
                   callbacks = [save_callbacks, early_stop_callbacks]
                   )



# =============================================================================
# # Fit model
# hist = unet.fit(X_train, y_train, 
#     batch_size=cf.BATCH_SIZE, 
#     epochs=cf.EPOCHS, 
#     #verbose=2, 
#     validation_data=(X_test, y_test),
#     #callbacks = [model_save_callbacks]
# )
# =============================================================================

e_time = datetime.now()

print("Training time: ", (e_time - s_time))

# Saving the model
unet.save("new_data_model.h5")

###############################################################################
print("----------------------------------------------------------------------")
print("############### Evaluation the training metrics ######################")
print("----------------------------------------------------------------------")


fig, s_plot = plt.subplots(1, 2, figsize=(20, 5))

s_plot[0].plot(results.history["loss"], color='r', label='train_loss')
s_plot[0].plot(results.history["val_loss"], color='b', label='validation_loss')
s_plot[0].set_title('Loss Comparison')
s_plot[0].legend()

s_plot[1].plot(results.history['accuracy'], color='r', label='training accuracy')
s_plot[1].plot(results.history['val_accuracy'], color='b', label='validation accuracy')
s_plot[1].set_title('Accuracy comparison')
s_plot[1].legend()
plt.show()



###############################################################################
# =============================================================================
# def check_predictions(index):
#     img = X_test[index]
#     img = img[np.newaxis, ...]
#     pred_y = unet.predict(img)
#     pred_mask = tf.argmax(pred_y[0], axis=-1)
#     pred_mask = pred_mask[..., tf.newaxis]
#     
#     fig, arr = plt.subplots(1,3, figsize=(15, 15))
#     arr[0].imshow(X_test[index])
#     arr[0].set_title('Raw image')
#     
#     arr[1].imshow(y_test[index])
#     arr[1].set_title('masked image')
#     
#     arr[2].imshow(pred_mask)
#     arr[2].set_title('predicted mask image')
#     cv2.imwrite('predicted_img.png', pred_mask)
#     
# index = 20
# check_predictions(index)    
# 
# =============================================================================


def check_predictions(path):
    
    img_raw = cv2.imread(path,1)
    img_raw = cv2.resize(img_raw, (cf.HEIGHT, cf.WIDTH))
    img = img_raw[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    fig, arr = plt.subplots(1,2, figsize=(15, 15))
    arr[0].imshow(img_raw)
    arr[0].set_title('Raw image')
    
    arr[1].imshow(pred_mask)
    arr[1].set_title('predicted mask image')
    cv2.imwrite('predicted_img.png', pred_mask)
    
check_predictions('00372.jpg')


###############################################################################
print("----------------------------------------------------------------------")
print("######################### Batch Prediction ###########################")
print("----------------------------------------------------------------------")

target_img_shape = (cf.HEIGHT, cf.WIDTH, 3)
target_label_shape = (cf.HEIGHT, cf.WIDTH, 1)

img_files, label_files = ut.load_data(cf.TEST_IMG, cf.TEST_LABEL)

# Take sample of every scene

X_test, y_test = ut.preprocess_data(img_files, label_files, target_img_shape, target_label_shape, cf.TEST_IMG, cf.TEST_LABEL)



y_pred1   = unet.predict(X_test, batch_size=1)
y_pred1_i = np.argmax(y_pred1, axis=3)
y_test1_i = np.argmax(y_test, axis=3)

print(y_test1_i.shape)

plt.imshow(y_pred1_i[0])
print(y_test1_i[0])
print(np.amax(y_test1_i[5]))
###############################################################################
print("----------------------------------------------------------------------")
print("############################## IoU ###################################")
print("----------------------------------------------------------------------")
ut.IoU(y_test[:,:,:,0], y_pred1_i)

# =============================================================================
# plt.imshow(X_test[0])
# 
# for i in range(X_test.shape[0]):
#     cv2.imwrite(os.path.join(cf.TEST_IMAGES_DIR, '{}.jpg'.format(i)), X_test[i])
#     cv2.imwrite(os.path.join(cf.TEST_LABELS_DIR, '{}.png'.format(i)), y_test[i])
# print(X_test.shape[0])
# print(y_test.shape)
# =============================================================================

y_pred=unet.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 10
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())





















