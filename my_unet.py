# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose, concatenate


def unit_encoder_block(block_input, n_filter=32, dropout_prob=0.3, max_pooling=True):
    en_layer = Conv2D(n_filter, 
                  kernel_size = 3,
                   activation = 'relu',
                   padding ='same',
                   kernel_initializer = 'HeNormal'
                  )(block_input)
    
    en_layer = Conv2D(n_filter, 
                  kernel_size = 3,
                   activation = 'relu',
                   padding ='same',
                   kernel_initializer = 'HeNormal'
                  )(en_layer)
    
    en_layer = BatchNormalization()(en_layer, training=False)
    
    # Reduce overfitting
    if dropout_prob > 0:
        en_layer = Dropout(dropout_prob)(en_layer)
        
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(en_layer)
    else:
        next_layer = en_layer
    
    skip_connection = en_layer
    
    return next_layer, skip_connection

def unit_decoder_block(block_input, skip_connection, n_filter=32):
    de_layer = Conv2DTranspose(n_filter,
                              kernel_size=(3,3),
                               strides=(2,2),
                              padding='same')(block_input)
    
    de_layer = concatenate([skip_connection, de_layer], axis=3)
    
    de_layer = Conv2D(n_filter,
                      kernel_size=3,
                      activation = 'relu',
                      padding='same',
                      kernel_initializer='HeNormal')(de_layer)
    
    de_layer = Conv2D(n_filter,
                  kernel_size=3,
                  activation = 'relu',
                  padding='same',
                  kernel_initializer='HeNormal')(de_layer)
    
    
    return de_layer

def u_net_model(input_shape=(128, 128, 3), n_filter=32, n_class=3):
    input_layer = Input(input_shape)
    
    en_block1 = unit_encoder_block(input_layer, n_filter, dropout_prob=0, max_pooling= True)
    en_block2 = unit_encoder_block(en_block1[0], n_filter*2, dropout_prob=0, max_pooling= True)
    en_block3 = unit_encoder_block(en_block2[0], n_filter*4, dropout_prob=0, max_pooling= True)
    en_block4 = unit_encoder_block(en_block3[0], n_filter*8, dropout_prob=0.3, max_pooling= True)
    en_block5 = unit_encoder_block(en_block4[0], n_filter*16, dropout_prob=0.3, max_pooling= False)
    
    de_block6 = unit_decoder_block(en_block5[0], en_block4[1], n_filter*8)
    de_block7 = unit_decoder_block(de_block6, en_block3[1], n_filter*4)
    de_block8 = unit_decoder_block(de_block7, en_block2[1], n_filter*2)
    de_block9 = unit_decoder_block(de_block8, en_block1[1], n_filter)
    
    conv10 = Conv2D(n_filter,
                 kernel_size=3,
                 activation='relu',
                 padding='same',
                   kernel_initializer='he_normal')(de_block9)
    
    conv11 = Conv2D(n_class, 1, padding='same')(conv10)
    
    # Define the model
    model = tf.keras.Model(inputs=input_layer, outputs=conv11)
    
    return model
    