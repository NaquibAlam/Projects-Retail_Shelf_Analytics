#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:19:36 2018

@author: Paritosh Pandey
"""

import os 
import re
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import style
style.use("ggplot")
import random
import tensorflow as tf
import pandas as pd
import sys
import json
import time, datetime, errno
from sklearn.cluster import KMeans
import pytesseract

from keras.models import model_from_json

class GetOutOfLoop(Exception):
    pass

class ShelfShare(object):
    
    graph_def = None
    loaded_model = None
    noOfPromo = None
    
    def __init__(self, path = '/data1/paritosh.pandey/model/frozen_inference_graph.pb',
                 model = "/data1/paritosh.pandey/model_31_32_2_sgd.json",
                 weights = "/data1/paritosh.pandey/weights_31_32_2_sgd.h5"):
    # def trial(self, path = '/data1/paritosh.pandey/model/frozen_inference_graph.pb',
    #              model = "/data1/paritosh.pandey/model_31_32_2_sgd.json",
    #              weights = "/data1/paritosh.pandey/weights_31_32_2_sgd.h5"):
        if self.graph_def == None:
            self._graph_load(path)
        if self.loaded_model == None:
            self._load_classifier(model, weights)
        if self.noOfPromo == None:
            self.noOfPromo = []
        
    def _graph_load(self, path):
        graph_path = path
        with tf.gfile.FastGFile(graph_path,'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
            # print('Detection module loaded...')
    
    def _load_classifier(self, model, weights):
        model_json = model
        arch_file_json = open(model_json,'r')
        model_json = json.load(arch_file_json)
        arch_file_json.close()
        self.loaded_model = model_from_json(model_json)
        self.loaded_model.load_weights(weights)
        # print('Classifier loaded...')
        
    def _image_load(self, path = '/data1/paritosh.pandey/CIGDataset/ShelfImages/C4_P06_N1_S4_1.JPG'):
        image_path = path
        img = Image.open(image_path)
        self.ing = np.array(img)
        self.ing = self.ing[:, :, ::-1].copy()
        # print('Image uploaded for processing...')
    
    def _find_object(self):
        
        self.imgCropTest = self.ing
        self.ing = cv.resize(self.ing,(600,600))
                
        self._out = self._sess.run([self._sess.graph.get_tensor_by_name('num_detections:0'),
            self._sess.graph.get_tensor_by_name('detection_scores:0'),
            self._sess.graph.get_tensor_by_name('detection_boxes:0'),
            self._sess.graph.get_tensor_by_name('detection_classes:0')],
           feed_dict={'image_tensor:0': self.ing.reshape(1, self.ing.shape[0], self.ing.shape[1], 3)})
        # print("Bounding boxes calculated...")
        self._sess.close()
        return self._out
    
    def _process_boxes(self, inputfolder):
        num_detections = int(self._out[0][0])
        #bound_box = self._out[2][0]
        bound_box = self._remove_overlap(self._out[2][0])
        bound_box = self._remove_overlap(bound_box, True)
        self._out[2][0] = bound_box
        #num_detections = bound_box[(bound_box[:,0] != 0) & (bound_box[:,1] != 0) & (bound_box[:,2] != 0) & (bound_box[:,3] != 0),:].shape[0]
        Y = np.zeros([num_detections,2])
        #Y[:,0] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),0]
        #Y[:,1] = bound_box[np.any([bound_box[:,0]!=0, bound_box[:,1]!=0],axis=0),2]
        Y = bound_box[0:num_detections,:]

        rack_info = self._find_racks(Y)
        self._draw_boxes(rack_info)
        results = self._classify_boxes(bound_box, rack_info)
        self._write_results(results, rack_info[1], inputfolder)
        return results
        
    def _remove_overlap(self, bound_box, flag=False):
        if not flag:
            indexS = np.argsort(bound_box[:,1])
        else:
            indexS = np.argsort(bound_box[:,3])
        indexSR = np.argsort(indexS)
        bound_box = bound_box[indexS,:]
        for i in range(2, bound_box.shape[0]):
            if (bound_box[i-1,0] == 0) & (bound_box[i-1,1] == 0) & (bound_box[i-1,2] == 0) & (bound_box[i-1,3] == 0):
                continue
            area1 = 1000000*(bound_box[i,3]-bound_box[i,1])*(bound_box[i,2]-bound_box[i,0])
            area2 = 1000000*(bound_box[i-1,3]-bound_box[i-1,1])*(bound_box[i-1,2]-bound_box[i-1,0])
            overlap_area = 1000000*(np.minimum(bound_box[i,3], bound_box[i-1,3])-np.maximum(bound_box[i,1], bound_box[i-1,1]))*(np.minimum(bound_box[i,2], bound_box[i-1,2])-np.maximum(bound_box[i,0], bound_box[i-1,0]))

            if overlap_area >= 0.8*np.minimum(area2,area1):
                if area2 > area1:
                    bound_box[i,:] = 0
                else:
                    bound_box[i-1,:] = 0
                
        return bound_box[indexSR,:]

    def _find_racks(self, Y):
        trial = Y[:,0]
        ind = np.argsort(trial)
        trial = np.sort(trial)
        diff = trial[1:]-trial[:-1]
        # diff2 = diff[1:]-diff[:-1]
    
        labInd = np.zeros(len(trial))
        for i in range(len(diff)):
            if diff[i] > np.max(diff)*0.3:
                labInd[i+1] = labInd[i]+1
            else:
                labInd[i+1] = labInd[i]
    
        col = [(255,0,0),(0,255,0),(0,0,255)]
        for i in range(3,len(np.unique(labInd))):
            col.append((255,random.randint(0,256),random.randint(0,256)))    
        # print('Rack information calculated...')
        return ind, labInd, col
    
    def _draw_boxes(self, rack_info):
        num_detections = int(self._out[0][0])
        rowsCT = self.imgCropTest.shape[0]
        colsCT = self.imgCropTest.shape[1]
        self.ing = cv.resize(self.ing, (colsCT,rowsCT))

        col = rack_info[2]
        labInd = rack_info[1]
        ind = rack_info[0]

        if (rowsCT >= 1500) or (colsCT >= 1500):
            thickness = 10
        elif (rowsCT >= 500) or (colsCT >= 500):
            thickness = 4 + (colsCT/1000 - 0.5)*6
        else:
            thickness = 4
                
        for i in range(num_detections):
            index = np.where(ind == i)
            score = float(self._out[1][0][i])
            bbox = [float(v) for v in self._out[2][0][i]]
            if (bbox[0] == 0) & (bbox[1] == 0) & (bbox[2] == 0) & (bbox[3] == 0):
                continue 
            if score > 0.3:
                x = bbox[1] * colsCT
                y = bbox[0] * rowsCT
                right = bbox[3] * colsCT
                bottom = bbox[2] * rowsCT
                cv.rectangle(self.ing, (int(x), int(y)), (int(right), int(bottom)), col[int(labInd[index[0][0]])], thickness=thickness)
    
        #cv.imshow('TensorFlow MobileNet-SSD', self.ing)

    def _draw_promo_boundary(self):
        # print("Running Contour Detection...")
        imgray = cv.cvtColor(self.ing_promo, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # print("Total number of contours:", len(contours))
        area_contours=[]

        for cnt in contours:
            area_contours.append(cv.contourArea(cnt))
        area_contours = np.array(area_contours)
        top_area_args = np.argsort(-area_contours)[:30]
        top_area_contours = area_contours[top_area_args]
        rect_xywh_txt = []

        for idx in top_area_args:
            cnt = contours[idx]
            x, y, w, h = cv.boundingRect(cnt)
            im_ocr = self.ing_promo[y:y+h, x:x+w]
            im_ocr_gray = cv.cvtColor(im_ocr, cv.COLOR_BGR2GRAY)
            _, im_ocr_thresh = cv.threshold(im_ocr_gray, 127, 255, 0)

            text = pytesseract.image_to_string(Image.fromarray(im_ocr_thresh))
            print(len(text))
            if(len(text) > 0):
                rect_xywh_txt.append([x,y,w,h])
        
        if len(rect_xywh_txt) == 0:
            return
        
        rect_xywh_sorted = sorted(rect_xywh_txt, key = lambda t:t[1])
        tolerance_y = int(self.ing_promo.shape[0]*0.06)
        rect_xywh_promos = []
        rect_xywh_promos.append([rect_xywh_sorted[0]])
        current_miny_rect = rect_xywh_sorted[0]
        idx = 0
        for rect in rect_xywh_sorted[1:]:
            if rect[1] < current_miny_rect[1] + tolerance_y:
                rect_xywh_promos[idx].append(rect)
            else:
                idx = idx+1
                rect_xywh_promos.append([rect])
                current_miny_rect = rect
                
        rect_xywh_promo_combined = []
        
        for rects in rect_xywh_promos:
            rects_arr = np.array(rects)
            n_rows = rects_arr.shape[0]
            right_arr = (rects_arr[:, 0] + rects_arr[:, 2]).reshape(n_rows, 1)
            bottom_arr= (rects_arr[:, 1] + rects_arr[:, 3]).reshape(n_rows, 1)
            rects_arr= np.hstack((rects_arr, right_arr, bottom_arr))
            
            min_arr = np.min(rects_arr, axis = 0)
            max_arr = np.max(rects_arr, axis=0)
            min_x, min_y = min_arr[0], min_arr[1]
            max_right, max_bottom = max_arr[4], max_arr[5]
            
            rect_xywh_promo_combined.append([min_x, min_y, max_right, max_bottom])

        rowsCT = self.imgCropTest.shape[0]
        colsCT = self.imgCropTest.shape[1]
        if (rowsCT >= 1500) or (colsCT >= 1500):
            thickness = 10
        elif (rowsCT >= 500) or (colsCT >= 500):
            thickness = 4 + (colsCT/1000 - 0.5)*6
        else:
            thickness = 4
            
        for xywh in rect_xywh_promo_combined:
            cv.rectangle(self.imgCropTest, (int(xywh[0]), int(xywh[1])), (int(xywh[2]), int(xywh[3])), (0,255,240), thickness=thickness)

        self.noOfPromo.append(len(rect_xywh_promo_combined))

        return
    
    def _classify_boxes(self, bound_box, rack_info):
        ind = rack_info[0]
        labInd = rack_info[1]
                
        rowsCT = self.imgCropTest.shape[0]
        colsCT = self.imgCropTest.shape[1]

        self.ing_promo = self.ing.copy()
        
        # print(rowsCT, colsCT)
        x_max = np.amax(bound_box[:,3])
        x_min = np.amin(bound_box[bound_box[:,1] > 0,1])

        shelfRatios = np.zeros((len(np.unique(labInd)),16))
        shelfCoverage = np.zeros((len(np.unique(labInd)),16))
        shelfStats = [None]*16
        boxClass = [None]*16
        length = np.zeros((len(np.unique(labInd)),1))
        infoList = []
        # include = [False]*16

        # for visibility index
        finalCoverData = []; placements = []; coverage = []; xInfo = [None]*16; vacInfo = [];

        size = np.ones((16,1))
        # print('Starting classification module...')
        noOfShelves = len(np.unique(labInd))
        for i in range(noOfShelves):
            index = np.where(labInd == i)
            bbox = bound_box[ind[index],:]

            # To remove zero size boxes
            if len(np.unique(bbox[:,0])) == 1:
                continue
            infoList.append(i)
            
            rowCoverDataA = []
            rowCoverDataB = []
            rowUncoverData = []
            rowPlacement = []

            # To remove boxes at the edges of the images
            bbox = np.sort(bbox, axis = 0)
            if 0 in np.unique(bbox):
                locs = np.unique(np.where(bbox == 0)[0])
                # print(locs)
                if len(np.unique(bbox[locs,:])) == 1:
                    bbox = bbox[np.unique(np.where(bbox != 0)[0]),:]
                else:
                    for loc in locs:
                        if len(np.where(bbox[loc, :] == 0)[0]) == 4:
                            bbox[loc,:] = np.nan
                    bbox = bbox[~np.any(np.isnan(bbox), axis=1)]

            maxX = 0
            minX = np.amin(bbox[:,1])
            images = np.zeros((bbox.shape[0],224,224,3)) 
            for j in range(bbox.shape[0]):
                testImg = self.imgCropTest[int(rowsCT*bbox[j,0]):int(rowsCT*bbox[j,2])+1,int(colsCT*bbox[j,1]):int(colsCT*bbox[j,3])+1]
                testImgR = cv.resize(testImg,(224,224))
                images[j,:,:,:] = testImgR
            pred_label = self.loaded_model.predict(images/255)

            if bbox[:,[1,3]].shape[0] < 3:
                cluster_info = [0]*(bbox[:,[1,3]].shape[0])
            else:
                cluster_info = KMeans(n_clusters = 3).fit_predict(bbox[:,[1,3]])
            
            lastLabel = 0
            newClusterLabels = []
            for cluster in range(len(cluster_info)):
                if cluster == 0:
                    newClusterLabels.append(lastLabel)
                else:
                    if cluster_info[cluster - 1] != cluster_info[cluster]:
                        lastLabel += 1
                    newClusterLabels.append(lastLabel)
            cluster_info = newClusterLabels

            # print("Cluster Info: ", cluster_info)

            y_min = int(np.min(bbox[:,0]*rowsCT))
            y_max = int(np.max(bbox[:,2]*rowsCT))
            self.ing_promo[y_min:y_max, :] = 0
            # print(pred_label, np.sum(pred_label, axis=1))

            #print(np.max(pred_label,1),np.argmax(pred_label,1))
            for j in range(bbox.shape[0]):
                # if np.max(pred_label[j]) < 0.3:
                #     continue
                if shelfStats[np.argmax(pred_label[j,:])] == None:
                    shelfStats[np.argmax(pred_label[j,:])] = []
                    boxClass[np.argmax(pred_label[j,:])] = []
                    xInfo[np.argmax(pred_label[j,:])] = [np.zeros((noOfShelves, 1)), np.zeros((noOfShelves, 1)), np.zeros((noOfShelves, 1))]
                    # include[np.argmax(pred_label[j,:])] = True

                shelfStats[np.argmax(pred_label[j,:])].append(bbox[j,3]-np.maximum(bbox[j,1],maxX))
                boxClass[np.argmax(pred_label[j,:])].append([i,j])
                shelfRatios[i,np.argmax(pred_label[j,:])] += 1
                shelfCoverage[i,np.argmax(pred_label[j,:])] += bbox[j,3]-np.maximum(bbox[j,1],maxX)

                rowPlacement.append(np.argmax(pred_label[j,:]))
                rowCoverDataA.append(bbox[j,3])
                rowCoverDataB.append(np.maximum(bbox[j,1], maxX))

                xInfo[np.argmax(pred_label[j,:])][cluster_info[j]][i] += 1
                
                maxX = bbox[j,3]
                size[np.argmax(pred_label[j,:])] = np.minimum(size[np.argmax(pred_label[j,:])],bbox[j,3]-bbox[j,1])

                if j == 0:
                    # print("xmin", infoList.index(i), bbox[j, 1], x_min)
                    vacInfo.append([infoList.index(i), bbox[j, 1] - x_min, -1, np.argmax(pred_label[j+1,:])])
                    # print("x", infoList.index(i), bbox[j+1, 1], bbox[j, 3])
                    vacInfo.append([infoList.index(i), bbox[j+1, 1] - bbox[j, 3], -1, np.argmax(pred_label[j+1,:])])
                elif j == bbox.shape[0]-1:
                    blankSpace = x_max - bbox[j, 3]
                    # print("xmax", infoList.index(i), x_max, bbox[j, 3])
                    vacInfo.append([infoList.index(i), x_max - bbox[j, 3], np.argmax(pred_label[j-1,:]), -1])
                else:
                    # print("x", infoList.index(i), bbox[j+1, 1], bbox[j, 3])
                    vacInfo.append([infoList.index(i), bbox[j+1, 1] - bbox[j, 3], np.argmax(pred_label[j-1,:]), np.argmax(pred_label[j+1,:])])

                # if bbox[j,1] > maxX:
                #     rowUncoverData.append()

            placements.append(rowPlacement)
            coverage.append([rowCoverDataA, rowCoverDataB])
            length[i] = 0.12*(x_max - x_min) + 0.88*(maxX - minX)
            # print("i ", length)

        self._draw_promo_boundary()

        # print('Shelf coverage calculated...')
        # print("All Shelves: ", xInfo)
        
        shelfRatios = shelfRatios[infoList,:]
        shelfCoverage = shelfCoverage[infoList,:]
        length = length[infoList,:]
        for i in range(16):
            if xInfo[i] != None:
                xInfo[i][0] = xInfo[i][0][infoList]
                xInfo[i][1] = xInfo[i][1][infoList]
                xInfo[i][2] = xInfo[i][2][infoList]
        # print("Selected Shelves: ", xInfo)

        weights = self._generate_weights(len(placements))
        heatBucket = np.zeros((17, 1))
        for i in range(len(weights)):
            S = 100 * weights[i] / np.sum(weights)
            for j in range(len(placements[i])):
                x2 = 1000*coverage[i][1][j]
                x1 = 1000*coverage[i][0][j]
                n = (2*length[i][0] - x1 - x2) * (x2 - x1)
                d = np.square(1000*length[i][0])
                heatBucket[placements[i][j]] += S*n/d
        # print(heatBucket, np.sum(heatBucket))
        # heatBucket = heatBucket / np.sum(heatBucket)
        heatBucket[16] = np.sum(heatBucket)
        heatBucket = heatBucket[heatBucket != 0]
        heatBucket[heatBucket < 0] = 0
        if heatBucket[-1:] > 100:
            heatBucket[-1:] = 100
            heatBucket[:-1] = 100*heatBucket[:-1]/np.sum(heatBucket[:-1])
        # print("HB:    ",heatBucket)

        # toReturn = self._process_for_data(shelfRatios, shelfCoverage, shelfStats, length, size, heatBucket, xInfo)
        toReturn = self._process_for_data(shelfRatios, shelfCoverage, shelfStats, length, vacInfo, heatBucket, xInfo, infoList)
        # for i in range(16):
        #     try:
        #         if toReturn[2][i] != None:
        #             toReturn[2][i][0] = toReturn[2][i][0][infoList]
        #             toReturn[2][i][1] = toReturn[2][i][1][infoList]
        #             toReturn[2][i][2] = toReturn[2][i][2][infoList]
        #     except ValueError as ex:
        #         pass

        locs = toReturn[4]
        for i in range(len(locs)):
            # print(locs[i])
            boxLoc = boxClass[locs[i][0]][locs[i][1]]
            # print(boxLoc)
            index = np.where(labInd == boxLoc[0])
            bbox = bound_box[ind[index],:]
            bbox = np.sort(bbox, axis = 0)
            finalBox = bbox[boxLoc[1],:]
            # print(finalBox)
            x = finalBox[1] * colsCT
            y = finalBox[0] * rowsCT
            right = finalBox[3] * colsCT
            bottom = finalBox[2] * rowsCT
            # print(int(x), int(y), int(right), int(bottom))
            cv.rectangle(self.ing, (int(x), int(y)), (int(right), int(bottom)), (255,255,255), thickness=8)

        return toReturn[0], toReturn[1], toReturn[2], toReturn[3], toReturn[5] 

    def _generate_weights(self, len):
        
        if len == 2:
            return [1, 1]
        elif len == 3:
            return [1,3,2]
        elif len == 4:
            return [2, 2.5, 2.5, 1]
        elif len == 5:
            return [1, 3, 5, 4, 2]

        return []

    def _max_find(self, one, two):
        if one < 3:
            maxOne = 0
        elif one <= 5:
            maxOne = 5
        elif one < 10:
            maxOne = 10
        else:
            maxOne = 15
        
        if two < 3:
            maxTwo = 0
        elif two <= 5:
            maxTwo = 5
        elif one < 10:
            maxTwo = 10
        else:
            maxTwo = 15

        return maxOne, maxTwo

    def _contest_compliance(self, one, two):
        maxOne, maxTwo = self._max_find(one, two)
        # print(maxOne, maxTwo)
        if maxOne == 0:
            return 2
        elif maxTwo == 0:
            return 1
        else:
            if ((one+1)/maxOne > 1) and ((two+1)/maxTwo > 1):
                return 0
            if ((one+1)/maxOne < 1) and ((two+1)/maxTwo < 1):
                if (one+1)/maxOne == (two+1)/maxTwo:
                    return 0
                elif (one+1)/maxOne > (two+1)/maxTwo:
                    return 1
                else:
                    return 2
            else:
                if ((one+1)/maxOne > 1):
                    return 2
                else:
                    return 1

    # def _process_for_data(self, shelfRatios, shelfCoverage, shelfStats, length, size, heatBucket, xInfo):
    def _process_for_data(self, shelfRatios, shelfCoverage, shelfStats, length, vacInfo, heatBucket, xInfo, infoList):

        # prodNames = ['Kent','Chesterfield','2000','Muratti','Monte_Carlo','Pall_Mall','LD','LM','Camel','Marlboro','Parliament','Lark','Lucky_Strike','Davidoff','Viceroy','Winston','West']
        prodNames = ['Kent','Chesterfield','2000','Muratti','MonteCarlo','PallMall','LM','LuckyStrike','Marlboro','Parliament','Lark','Other','Davidoff','Viceroy','Winston','West']

        shlfDf = pd.DataFrame(np.transpose(shelfRatios))
        shlfDf.set_index([prodNames], inplace = True)
        shlfDf.columns = ['Shelf_'+ str(i + 1) for i in range(shlfDf.shape[1])]
        shlfDf['Total'] = shlfDf.sum(axis = 1)
        shlfDf.Total.replace(0, np.nan, inplace = True)
        shlfDf.dropna(subset = ['Total'], axis = 0, inplace = True)
        shlfDf['Number_of_Products_(%)'] = 100*shlfDf.Total/shlfDf.Total.sum()

        df = pd.DataFrame(shelfCoverage)
        df.columns = prodNames
        df['Length Of Rack'] = length
        df = df.T
        df.columns = ['Shelf_' + str(i+1) for i in range(df.shape[1])]
        df['Occupied_Product_Area_(%)'] = df.sum(axis=1)
        df['Occupied_Product_Area_(%)'].replace(0,np.nan,inplace=True)
        df.dropna(subset=['Occupied_Product_Area_(%)'],axis=0,inplace=True)

        #statsDf = pd.DataFrame(index = [prodNames], columns = ["Mean", "Std. Dev", "Median", "Q1", "Q3"])
        #for j in range(len(shelfStats)):
           # prodStat = shelfStats[j]
            #if prodStat == None:
                #continue
            #statsDf.at[prodNames[j], "Mean"] = np.mean(prodStat)
            #statsDf.at[prodNames[j], "Std. Dev"] = np.std(prodStat)
            #statsDf.at[prodNames[j], "Median"] = np.median(prodStat)
            #statsDf.at[prodNames[j], "Q1"] = np.percentile(prodStat,25)
            #statsDf.at[prodNames[j], "Q3"] = np.percentile(prodStat,75)
        #statsDf.dropna(subset=['Mean'],axis=0,inplace=True)

        statsDf = pd.read_fwf('/data1/paritosh.pandey/stats/ShelfStatDF_354.txt')
        statsDf.set_index([statsDf['Unnamed: 0'].tolist()], inplace = True)
        statsDf.drop(columns = ['Unnamed: 0'], inplace = True)
        statsDf = statsDf.loc[df.index.tolist()[:-1]]

        vacancyComp = [None]*16
        # print(vacInfo)
        for i in range(len(vacInfo)):            
            stn = str(int(vacInfo[i][0]) + 1)
            # print(i)
            if vacInfo[i][2] == vacInfo[i][3]:
                width = statsDf['Mean'][prodNames[vacInfo[i][2]]] - 1.08*statsDf['Std. Dev'][prodNames[vacInfo[i][2]]]
                numberOfProdsPossible = np.floor(vacInfo[i][1]/width)
                # print("If Out: ", i, vacInfo[i][0], vacInfo[i][1], vacInfo[i][2], vacInfo[i][3], width, numberOfProdsPossible)
                if numberOfProdsPossible <= 0:
                    continue
                # print(vacancyComp[vacInfo[i][2]])
                try:
                    if vacancyComp[vacInfo[i][2]] == None:
                        vacancyComp[vacInfo[i][2]] = np.zeros((len(infoList), 1))
                except ValueError as ex:
                    pass
                
                vacancyComp[vacInfo[i][2]][vacInfo[i][0]] += numberOfProdsPossible
            else:
                # print("Else Out: ", vacInfo[i])
                # print('Shelf_'+stn, prodNames[vacInfo[i][2]])
                if (vacInfo[i][2] == -1) or (vacInfo[i][3] == -1):
                    if vacInfo[i][2] == -1:
                        prodOneCount = 0
                        prodTwoCount = shlfDf['Shelf_'+stn][prodNames[vacInfo[i][3]]]
                        width = [10, statsDf['Mean'][prodNames[vacInfo[i][3]]]]
                    else:
                        prodOneCount = shlfDf['Shelf_'+stn][prodNames[vacInfo[i][2]]]
                        prodTwoCount = 0
                        width = [statsDf['Mean'][prodNames[vacInfo[i][2]]], 10]
                else:
                    prodOneCount = shlfDf['Shelf_'+stn][prodNames[vacInfo[i][2]]]
                    prodTwoCount = shlfDf['Shelf_'+stn][prodNames[vacInfo[i][3]]]
                    width = [statsDf['Mean'][prodNames[vacInfo[i][2]]] - statsDf['Std. Dev'][prodNames[vacInfo[i][2]]], statsDf['Mean'][prodNames[vacInfo[i][3]]] - statsDf['Std. Dev'][prodNames[vacInfo[i][3]]]]

                # print(i, width, prodOneCount, prodTwoCount)
                
                blankSpace = vacInfo[i][1]
                flag = False

                try:
                    # print("Blankspace: ", blankSpace, width[0], width[1])
                    if (blankSpace < width[0]) and (blankSpace < width[1]):
                        raise Exception

                    while blankSpace > 0:
                        # print("Enter while for different boxes")
                        try:
                            # print("-_-")
                            if flag == False:
                                chosenProd = self._contest_compliance(prodOneCount, prodTwoCount)
                            # print("chosenIn: ",chosenProd)
                            
                            if chosenProd == 0:
                                chosenProd = 1

                            numberOfProdsPossible = np.floor(blankSpace/width[chosenProd-1])
                            if numberOfProdsPossible < 0:
                                numberOfProdsPossible = 0
                            # print("In: ", i, vacInfo[i][0], blankSpace, vacInfo[i][2], vacInfo[i][3], chosenProd, numberOfProdsPossible)

                            if numberOfProdsPossible == 0:
                                chosenProd = -chosenProd + 3
                                if flag == False:
                                    flag = True
                                    raise GetOutOfLoop
                                else:
                                    raise Exception 
                            if vacancyComp[vacInfo[i][1+chosenProd]] == None:
                                vacancyComp[vacInfo[i][1+chosenProd]] = np.zeros((df.shape[1], 1))

                            vacancyComp[vacInfo[i][1+chosenProd]][vacInfo[i][0]] += 1
                            blankSpace -= width[chosenProd-1]
                            if chosenProd == 1:
                                prodOneCount += 1
                            else:
                                prodTwoCount += 1

                        except GetOutOfLoop:
                            pass

                except Exception:
                    pass
        # print(vacancyComp)
        
        vacancyDf = pd.DataFrame([], ['Shelf_'+ str(i + 1) for i in range(shlfDf.shape[1]-2)], [])
        count = 0
        for j in range(len(vacancyComp)):
            try:
                if vacancyComp[j] == None:
                    count += 1
                    continue
            except ValueError as ex:
                vacancyDf[prodNames[j]] = vacancyComp[j]#[infoList]
                pass
        # print(vacancyDf)
        if count == 16:
            vacancyDf[shlfDf.index.tolist()[0]] = np.zeros((shlfDf.shape[1]-2,1))


        quantFail = [None]*statsDf.shape[0]
        lenFail = [None]*statsDf.shape[0]
        locs = []

        horzDf = pd.DataFrame([], ["Left", "Middle", "Right"], [])
        for j in range(len(prodNames)):
            if xInfo[j] != None:
                horzDf[prodNames[j]] = xInfo[j]
        # print(horzDf)

        for j in range(statsDf.shape[0]):
            prodStat = shelfStats[prodNames.index(statsDf.index.tolist()[j])]
            #print("Product: ", statsDf.index.tolist()[j])
            #print("Data from Image: ", prodStat)
            #print("Overall Data: ", statsDf.iloc[j,:])
            iqf = statsDf.iloc[j, 1] #- statsDf.iloc[j, 3]
            #print('IQF: ', iqf)
            for i in range(len(prodStat)):
                if (prodStat[i] < statsDf.iloc[j,0] - 3*iqf) or (prodStat[i] > statsDf.iloc[j,0] + 3*iqf):
                    locs.append([prodNames.index(statsDf.index.tolist()[j]),i])
                    
            pQLess = sum(a < statsDf.iloc[j,0] - 3*iqf for a in prodStat)
            pQMore = sum(a > statsDf.iloc[j,0] + 3*iqf for a in prodStat)
            #print(pQLess, pQMore)
            quantFail[j] = pQLess + pQMore
            
            pSLess = sum(a for a in prodStat if a < statsDf.iloc[j,0] - 3*iqf)
            pSMore = sum(a for a in prodStat if a > statsDf.iloc[j,0] + 3*iqf)
            lenFail[j] = pSLess + pSMore

        quantFail = [a for a in quantFail if a != None]
        quantFail.append(shlfDf.Total.sum())
        lenFail = [a for a in lenFail if a != None]
        lenFail.append(df.iloc[df.shape[0]-1, df.shape[1]-1])
        
        statsDf = pd.DataFrame([quantFail, lenFail])
        columns = shlfDf.index.tolist()
        columns.append("Total")
        statsDf.columns = columns
        statsDf = statsDf.T
        statsDf.columns = ["Shelf_1", "Shelf_2"]
        statsDf["Heat"] = heatBucket
        
        # size = size[size != 1]

        # return shlfDf, df, size, statsDf, locs, horzDf
        return shlfDf, df, vacancyDf, statsDf, locs, horzDf
    
    def _write_results(self, results, labInd, inputfolder):
        # result = Image.fromarray(self.ing)
        # if self.ing.shape[0] > 1200:
            # rows = 1200
            # cols = int(1200*self.ing.shape[1]/self.ing.shape[0])
        # else:
        rows = self.ing.shape[0]
        cols = self.ing.shape[1]

        if rows > 1200:
            aspectRatio = rows/cols
            cols = 1200
            rows = int(cols/aspectRatio)


        fullpath = os.path.join('/data1/paritosh.pandey/static/results', inputfolder)

        # resultC = Image.fromarray(cv.resize(self.ing,(rows,cols)))
        r = re.compile(r'result_\d+.jpg$')
        l = []
        for f in os.listdir(fullpath):
            if not os.path.isfile(os.path.join(fullpath, f)): continue;
            # print(f)
            if os.path.isfile(os.path.join(fullpath, f)) & (f[0:7]=='result_'): 
                l = f;
                # print('So So True ', True)

        if len(l) == 0:
            end = 1
        else:
            end = int(l[7:-4])+1
        # print(end)

        font = cv.FONT_HERSHEY_SIMPLEX
        txt = ""
        listOfIndices = results[0].index.tolist()
        rowsCT = self.imgCropTest.shape[0]
        colsCT = self.imgCropTest.shape[1]
        if (rowsCT >= 1500) or (colsCT >= 1500):
            thickness = 10
            inc = 70
        elif (rowsCT >= 500) or (colsCT >= 500):
            thickness = 5 + (colsCT/1000 - 0.5)*5
            inc = 20 + (colsCT/1000 - 0.5)*50
        else:
            thickness = 5
            inc = 20

        y = rowsCT - 50
        for i in range(results[0].shape[0]):
            txt = "{}: {}".format(listOfIndices[i], results[0]['Total'][i])
            cv.putText(self.ing, txt, (50, y), font, 2, (255,255,255), 4)
            y -= inc
        txt = "{}: {}".format('Total', results[0]['Total'].sum())
        cv.putText(self.ing, txt, (50, y), font, 2, (255,255,255), 4)
            
        
        cv.imwrite(fullpath + '/result_' + str(end) + '.jpg', cv.resize(self.ing, (rows, cols)))
        cv.imwrite(fullpath + '/resultP_' + str(end) + '.jpg', cv.resize(self.imgCropTest, (rows, cols)))
        # cv.imwrite('/data1/paritosh.pandey/static/resultC_'+str(end)+'.jpg', cv.resize(self.ing,(rows,cols)))
        # result.save('/data1/paritosh.pandey/static/result_'+str(end)+'.jpg')
        # resultC.save('/data1/paritosh.pandey/static/resultC_'+str(end)+'.jpg')
        # print('Resulting image saved...')
        
        df = results[1]
        # size = results[2]
        vacancy = results[2]

        with open(fullpath + '/ProdCount_' + str(end) + '.txt', 'w') as f:
           f.write(round(results[0]).to_string())
        with open(fullpath + '/ProdHorz_' + str(end) + '.txt', 'w') as f:
           f.write(round(results[4].T).to_string())
        with open(fullpath + '/ProdShare_' + str(end) + '.txt', 'w') as f:
           occupy = df["Occupied_Product_Area_(%)"]
           occupy = occupy[-1:]
           # print(occupy.to_string()) 
           resultDf = round(100*df/df.iloc[df.shape[0]-1])
           resultDf = resultDf.iloc[[i for i in range(resultDf.shape[0]-1)]]
           resultDf["Length"] = occupy[0]
           f.write(resultDf.to_string())

        # print(vacancy)

        if vacancy.T.shape[0] != 0:
            with open(fullpath + '/ProdVacancy_' + str(end) + '.txt', 'w') as f:
                # vacancy = np.zeros((resultDf.shape[0], resultDf.shape[1]-2))

                # lengthOfRacks = df.iloc[df.shape[0]-1]
                # for i in range(df.shape[1]-1):
                #     st = 'Shelf_' + str(i+1)
                #     if resultDf[st].sum() < 99.9:
                #         space = (100 - resultDf[st].sum())*lengthOfRacks[st]/100
                #         vacancy[:,i] = space/size
                # vacancy[(vacancy < 1) & (vacancy > 0.95)] = 1
                # vacancy = np.floor(vacancy)

                # vacDf = pd.DataFrame(vacancy)
                # vacDf.set_index(results[0].index,inplace=True)

                # vacDf.columns = ['Shelf_' + str(i+1) for i in range(df.shape[1]-1)]

                f.write(vacancy.T.to_string())
               
               #f.write('Rackwise Shelf Share:\n')
               # for i in range(len(np.unique(labInd))):
               #     val = 100*df.loc[i].values/df.loc[i,'sum']
               #     f.write('Rack ' + str(i) + '\n' + str(val[:-1]) + '\n')
               #     f.write(str(np.sum(val[:-1])) + '\n')
               # val = 100*df.loc[len(np.unique(labInd))].values/df.loc[len(np.unique(labInd)),'sum']
               # f.write('\nOverall Shelf Share: \n'+ str(val[:-1]) + '\n')
               # f.write(str(np.sum(val[:-1])))

        with open(fullpath + '/ProdSkew_' + str(end) + '.txt', 'w') as f:
           f.write(round(results[3], 2).to_string())

        # print('Shelf-Share calculated and saved...\n')
        
    def main(self, inputfile, inputfolder):
        # self.trial()
        self._image_load(inputfile)
        self._sess = tf.Session()
        self._sess.graph.as_default()
        tf.import_graph_def(self.graph_def, name='')
        
        self._find_object()
        dataTable = self._process_boxes(inputfolder)
        return dataTable
        
def main_fn(inputfolder):
    shelf = ShelfShare()
    # for f in os.listdir('static'):
    #     if os.path.isfile(os.path.join('static', f)):
    #         os.remove(os.path.join('static',f))

    fullpath = os.path.join('/data1/paritosh.pandey/static/results', inputfolder)
    try:
        os.mkdir(fullpath)
    except OSError as exc:
        try:
            os.mkdir('/data1/paritosh.pandey/static/results')
            os.mkdir(fullpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass

        if exc.errno != errno.EEXIST:
            raise
        pass

    path = os.path.join('/data1/paritosh.pandey/static/temp', inputfolder)
    noOfShelves = 0
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f[0:7] == "result_":
                table = shelf.main(os.path.join(path, f), inputfolder)
                noOfShelves += 1

    with open(fullpath + '/timeStamp.txt', 'w') as f:
        f.write(str(datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y %H:%M:%S')) + "\n"+inputfolder + "\nABC\n" + str(noOfShelves) + "\nPromo " + str(shelf.noOfPromo))

    return table

# if __name__=='__main__':
#     main_fn(os.path.join('/data1/paritosh.pandey/static/temp', inputfolder))

#main_fn(sys.argv[1])
