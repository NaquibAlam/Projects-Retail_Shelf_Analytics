#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:58:21 2018

@author: Naquib
"""

import os, sys, math, shutil, PIL, json
import numpy as np
import tensorflow as tf
import scipy.io as sio
import keras 
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from collections import Counter

def train_test_img_split(base_dir,dosplit=False):
    if dosplit is False:
        return
    #print("returned_2")
    folders=os.listdir(base_dir)
    folders=sorted(np.array(folders).astype(int))
    folders=np.array(folders).astype(str)
    #print(folders)
    train_img=[]
    val_img=[]
    test_img=[]
    train_test_img_path=[]
    for x in folders:
        class_files=os.listdir(os.path.join(base_dir,x))
        len_class_files=len(class_files)
        #file_sel_idx=np.random.shuffle(np.arange(len_class_files)
        end_train_idx=int(len_class_files*0.85)
        end_val_idx=int(len_class_files*0.95)
        #train_files=class_files[:end_train_idx]
        #test_files=class_files[end_train_idx:]
        train_files=class_files[:end_train_idx]
        val_files=class_files[end_train_idx:end_val_idx]
        test_files=class_files[end_val_idx:]
        train_img.append(train_files)
        val_img.append(val_files)
        test_img.append(test_files)
        train_test_img_path.append(os.path.join(base_dir,x))
    return(train_test_img_path,train_img,val_img,test_img,folders)
    
def create_train_val_test_dir(train_test_img_src_path,base_dir_split,train_img,val_img,test_img,create_dir=False):
    if create_dir is False:
        return
    #print("Returned_2")
    dirs=["train","val","test"]
    for i,x in enumerate(dirs):
        if (os.path.exists(os.path.join(base_dir_split,x))) is False:
            os.mkdir(os.path.join(base_dir_split,x))
        if i==0:
            for j,y in enumerate(train_img):
                if (os.path.exists(os.path.join(base_dir_split,x,str(j)))) is False:
                    os.mkdir(os.path.join(base_dir_split,x,str(j)))
                for z in y:
                    shutil.copy(os.path.join(train_test_img_src_path[j],z),os.path.join(base_dir_split,x,str(j)))
        elif i==1:
            for j,y in enumerate(val_img):
                if (os.path.exists(os.path.join(base_dir_split,x,str(j)))) is False:
                    os.mkdir(os.path.join(base_dir_split,x,str(j)))
                for z in y:
                    shutil.copy(os.path.join(train_test_img_src_path[j],z),os.path.join(base_dir_split,x,str(j)))
        else:
            for j,y in enumerate(test_img):
                if (os.path.exists(os.path.join(base_dir_split,x,str(j)))) is False:
                    os.mkdir(os.path.join(base_dir_split,x,str(j)))
                for z in y:
                    shutil.copy(os.path.join(train_test_img_src_path[j],z),os.path.join(base_dir_split,x,str(j)))

def get_number_images(base_dir):
    files=["train","valid","test"]
    n_images=[0,0,0]
    for i,x in enumerate(files):
        folders=os.listdir(os.path.join(base_dir,x))
        for y in folders:
            class_image=os.listdir(os.path.join(base_dir,x,y))
            n_images[i]+=len(class_image)
    return n_images      
       
def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    print("COUNTER: ",counter)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}

def fine_tune_dense(base_dir_split,n_images,do_train=False):
    if do_train is False:
        return
    summ_file=open('model_summary.txt','w')
    train_dir=os.path.join(base_dir_split,"train")
    val_dir=os.path.join(base_dir_split,"valid")
    #test_dir=os.path.join(base_dir_split,"test")
    img_width, img_height = 224,224
    n_class=16
    batch_size=32
    epochs=20
    learning_rate=0.001
    decay_rate=1/epochs
    nb_train_samples=n_images[0]
    nb_validation_samples=n_images[1]
    step_per_epoch=nb_train_samples/batch_size
    validation_step= nb_validation_samples/batch_size
    model = applications.VGG16(weights='imagenet', include_top=False,input_shape = (224,224,3))
    print('Model loaded.')
    #return model
    
    top_model = Sequential()
    for layer in model.layers:
        top_model.add(layer)
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #return model
    top_model.add(Dense(256, activation='relu',kernel_initializer='he_normal',bias_initializer=keras.initializers.Constant(value=0.1)))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(n_class, activation='softmax',kernel_initializer='he_normal',bias_initializer=keras.initializers.Constant(value=0.1)))
    #model.add(top_model)
    print('Top dense layer added')
    #return top_model
    for layer in top_model.layers[:19]:
        layer.trainable = False
    sgd=optimizers.SGD(lr=learning_rate, momentum=0.9, decay=decay_rate)
    top_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
    train_dir,target_size=(img_height, img_width),batch_size=batch_size,class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
    val_dir,target_size=(img_height, img_width),batch_size=batch_size, class_mode='categorical')
    #print("Train_Class_indices:",train_generator.class_indices)
    #print("Val_Class_Indices:",validation_generator.class_indices)
    class_weight=get_class_weights(train_generator.classes,0.1)
    #print("CLASS_WEIGHTS: ",class_weight)
    checkpoint_path='CIGDataset/SplitData/models/KerasTest/VGG16_Adam_30_0.hdf5'
    checkpoint= ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list=[checkpoint]  
    hist=top_model.fit_generator(
    train_generator,
    steps_per_epoch=step_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_step,
    class_weight=class_weight,
    callbacks= callback_list)
    
    model_json=top_model.to_json()
    with open("CIGDataset/SplitData/models/KerasTest/VGG16_Adam_30_0.json","w") as json_file:
        json.dump(model_json,json_file)
    top_model.save_weights("CIGDataset/SplitData/models/KerasTest/VGG16_Adam_30_0.h5") 
    with open('CIGDataset/SplitData/models/KerasTest/VGG16_Hist_Adam_30_0.json',"w") as json_file:
        json.dump(hist.history,json_file)
    print(top_model.summary(),file=summ_file)
    summ_file.close()
    return top_model

def do_predict(model_json,weights,base_dir_split,n_images,dopredict=False):
    if dopredict is False:
        return
    test_dir=os.path.join(base_dir_split,"test")
    nb_test_samples=n_images[2]
    #nb_test_samples=7724
    img_height,img_width=224,224
    batch_size=32
    arch_file_json=open(model_json,'r')
    model_json=arch_file_json.load()
    arch_file_json.close()
    loaded_model=model_from_json(model_json)
    loaded_model.load_weights(weights)
    test_datagen=ImageDataGenerator(rescale=1. / 255)
    test_generator=test_datagen.flow_from_directory(test_dir,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical',shuffle=False)
    print("test class indices: ",test_generator.class_indices)
    #print("test classes: ",test_generator.classes)
    pred_label=loaded_model.predict_generator(test_generator,int(math.ceil(nb_test_samples/batch_size))+1)
    print("Shape: ",pred_label.shape)
    y_true=test_generator.classes
    y_pred=np.argmax(pred_label,1)
    print(confusion_matrix(y_true,y_pred))
    #print(np.max(pred_label,1),np.argmax(pred_label,1))
    return pred_label
    
if __name__ == "__main__":
    base_dir_whole= r"/data1/naquib.alam/CIGDataset/CroppedImages"
    base_dir_split= r"/data1/naquib.alam/CIGDataset/SplitData"
    #print(base_dir_whole,base_dir_split)
    #x="/data1/naquib.alam/ShelfMonitoring/ObjectDetection/CiggarateDataset/grocerydataset/CroppedImageCategoryRefined/WholeData"
    #print("Trax",type(base_dir_whole),x==base_dir_whole)
    #train_test_img_src_path,train_img,val_img,test_img,folders = train_test_img_split(base_dir_whole)
    #print(type(train_test_img_src_path),type(train_img),type(val_img),type(test_img))
    #create_train_val_test_dir(train_test_img_src_path,base_dir_split,train_img,val_img,test_img)
    n_images = get_number_images(base_dir_split)
    print(n_images)
    model = fine_tune_dense(base_dir_split,n_images,True)
    #pred_label=do_predict("model_30_32_LIT.json","weights_30_32_LIT.h5",base_dir_split,n_images,True)
    #print(np.argmax(pred_label,1))  
