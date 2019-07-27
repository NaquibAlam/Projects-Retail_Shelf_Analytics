import os 
import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import json

from keras.models import model_from_json

def _remove_overlap(bound_box, flag = False):
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
            if area2 < area1:
                bound_box[i,:] = 0
            else:
                bound_box[i-1,:] = 0
            
    return bound_box[indexSR,:]

## Object Detection Graph
graph_path = '/data1/paritosh.pandey/model/frozen_inference_graph.pb'
with tf.gfile.FastGFile(graph_path,'rb') as g:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(g.read())
    print('Detection module loaded...')

mainG = tf.get_default_graph()

## Classifier
model_json = "/data1/paritosh.pandey/model_31_32_2_sgd.json"
weights = "/data1/paritosh.pandey/weights_31_32_2_sgd.h5"
arch_file_json = open(model_json,'r')
model_json = json.load(arch_file_json)
arch_file_json.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights(weights)
print('Classifier loaded...')

imagePath = "/data1/paritosh.pandey/CIGDataset/ShelfImages"

shelfStats = [None]*16
include = [False]*16

ui = 0
sess = tf.Session()
tf.import_graph_def(graph_def, name='')
for f in os.listdir(imagePath):
	print('\n')

	# if ui%5 == 0:
	# 	print(ui)
	
	## Load Image
	image_path = os.path.join(imagePath, f)
	img = Image.open(image_path)
	ing = np.array(img)
	ing = ing[:, :, ::-1].copy()
	print('Image uploaded for processing... ', f)

	## Object Detection
	imgCropTest = ing
	ing = cv.resize(ing, (600, 600))

	#with tf.Session() as sess:
		# sess.graph.as_default()
		#tf.import_graph_def(graph_def, name='')
	out = sess.run([mainG.get_tensor_by_name('num_detections:0'),
            mainG.get_tensor_by_name('detection_scores:0'),
            mainG.get_tensor_by_name('detection_boxes:0'),
            mainG.get_tensor_by_name('detection_classes:0')],
           feed_dict={'image_tensor:0': ing.reshape(1, ing.shape[0], ing.shape[1], 3)})
	print("Bounding boxes calculated...")
		#sess.close()

	num_detections = int(out[0][0])
	bound_box = _remove_overlap(out[2][0])
	bound_box = _remove_overlap(bound_box, True)
	out[2][0] = bound_box
	Y = np.zeros([num_detections,2])
	Y = bound_box[0:num_detections,:]

	## Find Racks
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

	print('Rack information calculated...')

	## Classify Boxes

	rowsCT = imgCropTest.shape[0]
	colsCT = imgCropTest.shape[1]
	x_max = np.amax(bound_box[:,3])
	x_min = np.amin(bound_box[bound_box[:,1] > 0,1])

	size = np.ones((16,1))
	print('Starting classification module...')
	for i in range(len(np.unique(labInd))):
		index = np.where(labInd == i)
		bbox = bound_box[ind[index],:]

		# To remove zero size boxes
		if len(np.unique(bbox[:,0])) == 1:
			continue

		# To remove boxes at the edges of the images
		bbox = np.sort(bbox, axis = 0)
		if 0 in np.unique(bbox):
			locs = np.unique(np.where(bbox == 0)[0])
			if len(np.unique(bbox[locs,:])) == 1:
				bbox = bbox[np.unique(np.where(bbox != 0)[0]),:]
			else:
				for loc in locs:
					if len(np.where(bbox[loc, :] == 0)[0]) == 4:
						bbox[loc,:] = np.nan
				bbox = bbox[~np.any(np.isnan(bbox), axis = 1)]

		maxX = 0
		minX = np.amin(bbox[:,1])
		images = np.zeros((bbox.shape[0],224,224,3))
		for j in range(bbox.shape[0]):
			testImg = imgCropTest[int(rowsCT*bbox[j,0]):int(rowsCT*bbox[j,2])+1,int(colsCT*bbox[j,1]):int(colsCT*bbox[j,3])+1]
			testImgR = cv.resize(testImg,(224,224))
			images[j,:,:,:] = testImgR
		print("Classify Row ", i)
		pred_label = loaded_model.predict(images/255)

		#print(np.max(pred_label,1),np.argmax(pred_label,1))
		for j in range(bbox.shape[0]):
			if shelfStats[np.argmax(pred_label[j,:])] == None:
				shelfStats[np.argmax(pred_label[j,:])] = []
				include[np.argmax(pred_label[j,:])] = True

			shelfStats[np.argmax(pred_label[j,:])].append((bbox[j,3]-bbox[j,1])*(bbox[j,2]-bbox[j,0]))
	print('Shelf coverage calculated...')
	ui+=1

	# if ui%20 == 0:
	# 	print("Data Processing")
	# 	prodNames = ['Kent','Chesterfield','2000','Muratti','MonteCarlo','PallMall','LM','LuckyStrike','Marlboro','Parliament','Lark','Other','Davidoff','Viceroy','Winston','West']
	# 	print('shelfStats: \n',shelfStats)

	# 	statsDf = pd.DataFrame(index = [prodNames], columns = ["Mean", "Std. Dev", "Median", "Q1", "Q3"])
	# 	for j in range(len(shelfStats)):
	# 		prodStat = shelfStats[j]
	# 		if prodStat == None:
	# 			continue
	# 		statsDf.at[prodNames[j], "Mean"] = np.mean(prodStat)
	# 		statsDf.at[prodNames[j], "Std. Dev"] = np.std(prodStat)
	# 		statsDf.at[prodNames[j], "Median"] = np.median(prodStat)
	# 		statsDf.at[prodNames[j], "Q1"] = np.percentile(prodStat,25)
	# 		statsDf.at[prodNames[j], "Q3"] = np.percentile(prodStat,75)
	# 	statsDf.dropna(subset=['Mean'],axis=0,inplace=True)
	# 	print('statsDf: \n', statsDf)

	# 	with open('/data1/paritosh.pandey/stats/ShelfStatDF_'+str(ui)+'.txt','w') as f:
	# 		f.write(statsDf.to_string())

# with open('/data1/paritosh.pandey/ShelfStat.txt','w') as f:\
# 	f.write(shelfStats)

## Process for Data
print("Data Processing")
prodNames = ['Kent','Chesterfield','2000','Muratti','MonteCarlo','PallMall','LM','LuckyStrike','Marlboro','Parliament','Lark','Other','Davidoff','Viceroy','Winston','West']

statsDf = pd.DataFrame(index = [prodNames], columns = ["Mean", "Std. Dev", "Median", "Q1", "Q3"])
for j in range(len(shelfStats)):
	prodStat = shelfStats[j]
	if prodStat == None:
		continue
	statsDf.at[prodNames[j], "Mean"] = np.mean(prodStat)
	statsDf.at[prodNames[j], "Std. Dev"] = np.std(prodStat)
	statsDf.at[prodNames[j], "Median"] = np.median(prodStat)
	statsDf.at[prodNames[j], "Q1"] = np.percentile(prodStat,25)
	statsDf.at[prodNames[j], "Q3"] = np.percentile(prodStat,75)
statsDf.dropna(subset=['Mean'],axis=0,inplace=True)

with open('/data1/paritosh.pandey/stats/ShelfStatDFArea_'+str(ui)+'.txt','w') as f:
	f.write(statsDf.to_string())
