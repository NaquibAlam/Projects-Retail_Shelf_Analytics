#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:50:51 2018

@author: fractaluser
"""

import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import style
style.use("ggplot")
import random
import tensorflow as tf
import pandas as pd
import time

from keras.models import model_from_json

graph_path = '/data1/paritosh.pandey/model/frozen_inference_graph.pb'
with tf.gfile.FastGFile(graph_path,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print('Object detection module loaded...')
    
name = 'C4_P06_N1_S4_1.JPG'
image_path = '/data1/paritosh.pandey/CIGDataset/ShelfImages/' + name
img = Image.open(image_path)
ing = np.array(img)
ing = ing[:, :, ::-1].copy()

imgCropTest = ing
rowsCT = imgCropTest.shape[0]
colsCT = imgCropTest.shape[1]

ing = cv.resize(ing,(600,600))
rows = ing.shape[0]
cols = ing.shape[1]
print('Image data acquired...')

with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')],
           feed_dict={'image_tensor:0': ing.reshape(1, ing.shape[0], ing.shape[1], 3)})
    print("Bounding boxes calculated...")
    
num_detections = int(out[0][0])
bound_box = out[2][0]
Y = np.zeros([num_detections,3])
Y[:,0] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),0]
Y[:,1] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),2]
    
X = np.zeros([num_detections,2])
X[:,0] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),1]
X[:,1] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),3]

trial = Y[:,0]
ind = np.argsort(trial)
trial = np.sort(trial)
diff = trial[1:]-trial[:-1]
diff2 = diff[1:]-diff[:-1]

labInd = np.zeros(len(trial))
for i in range(len(diff)):
    if diff[i] > np.max(diff)*0.3:
        labInd[i+1] = labInd[i]+1
    else:
        labInd[i+1] = labInd[i]

col = [(255,0,0),(0,255,0),(0,0,255)]
for i in range(3,len(np.unique(labInd))):
    col.append((255,random.randint(0,256),random.randint(0,256)))
print('Rack limits calculated...')

for i in range(num_detections):
    index = np.where(ind == i)
    score = float(out[1][0][i])
    bbox = [float(v) for v in out[2][0][i]]
    if score > 0.3:
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        cv.rectangle(ing, (int(x), int(y)), (int(right), int(bottom)), col[int(labInd[index[0][0]])], thickness=2)

#cv.imshow('TensorFlow MobileNet-SSD', ing)
result = Image.fromarray(ing)
result.save('/data1/paritosh.pandey/result.jpg')
print('Resulting image saved...')

model_json = "/data1/paritosh.pandey/model_30_32_cropped.json"
weights = "/data1/paritosh.pandey/weights_30_32_cropped.h5"
arch_file_json = open(model_json,'r')
model_json = arch_file_json.read()
arch_file_json.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights(weights)
print('Classification module loaded...')

x_max = np.amax(bound_box[:,3])
x_min = np.amin(bound_box[bound_box[:,1] > 0,1])

shelfRatios = np.zeros((len(np.unique(labInd)),17))
shelfCoverage = np.zeros((len(np.unique(labInd)),17))
length = np.zeros((len(np.unique(labInd)),1))
t0 = time.time()
for i in range(len(np.unique(labInd))):
    index = np.where(labInd == i)
    bbox = bound_box[ind[index],:]
    bbox = np.sort(bbox, axis = 0)
    maxX = 0
    minX = np.amin(bbox[:,1])
    images = np.zeros((bbox.shape[0],224,224,3)) 
    for j in range(bbox.shape[0]):
        testImg = imgCropTest[int(rowsCT*bbox[j,0]):int(rowsCT*bbox[j,2])+1,int(colsCT*bbox[j,1]):int(colsCT*bbox[j,3])+1]
        testImgR = cv.resize(testImg,(224,224))
        images[j,:,:,:] = testImgR
    pred_label=loaded_model.predict(images/255)
    #print(np.max(pred_label,1),np.argmax(pred_label,1))
    for j in range(bbox.shape[0]):
        shelfRatios[i,np.argmax(pred_label[j,:])] += 1
        shelfCoverage[i,np.argmax(pred_label[j,:])] += bbox[j,3]-np.maximum(bbox[j,1],maxX)
        maxX = bbox[j,3]
    length[i] = 0.12*(x_max - x_min) + 0.88*(maxX - minX)
print('Shelf coverage calculated...')
print('Time for bottleneck: ', time.time() - t0)

df = pd.DataFrame(shelfCoverage)
df['sum'] = length
df = df.append(df.sum(axis=0),ignore_index=True)

with open('result.txt','w') as f:
   f.write('Shelf Count for products:\n' + str(shelfRatios) + '\n')
   f.write('Rackwise Shelf Share:\n')
   for i in range(len(np.unique(labInd))):
       val = 100*df.loc[i].values/df.loc[i,'sum']
       f.write('Rack ' + str(i) + '\n' + str(val[:-1]) + '\n')
       f.write(str(np.sum(val[:-1])) + '\n')
   val = 100*df.loc[len(np.unique(labInd))].values/df.loc[len(np.unique(labInd)),'sum']
   f.write('\nOverall Shelf Share: \n'+ str(val[:-1]) + '\n')
   f.write(str(np.sum(val[:-1])))
print('Shelf share written in file result.txt...')
