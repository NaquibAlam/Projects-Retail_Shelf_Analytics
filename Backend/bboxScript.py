import pandas as pd
import cv2 as cv
from PIL import Image
import numpy as np
import tensorflow as tf

df = pd.read_csv('/data1/paritosh.pandey/data/test_labels.csv')

with tf.gfile.FastGFile('/data1/paritosh.pandey/model/frozen_inference_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sess = tf.Session()
sess.graph.as_default()
tf.import_graph_def(graph_def, name='')

for name in df.filename.unique():
    #name = df.filename[0]
    # Read and preprocess an image.
    img = Image.open('/data1/paritosh.pandey/CIGDataset/ShelfImages/'+name)
    inp = np.array(img)
    inp = inp[:, :, ::-1].copy()
    rows = inp.shape[0]
    cols = inp.shape[1]
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                sess.graph.get_tensor_by_name('detection_scores:0'),
                sess.graph.get_tensor_by_name('detection_boxes:0'),
                sess.graph.get_tensor_by_name('detection_classes:0')],
               feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    num_detections = int(out[0][0])
    
    for i in range(num_detections):
	classId = int(out[3][0][i])
    	score = float(out[1][0][i])
    	bbox = [float(v) for v in out[2][0][i]]
    	if score > 0.3:
      	   x = bbox[1] * cols
           y = bbox[0] * rows
           right = bbox[3] * cols
           bottom = bbox[2] * rows
           cv.rectangle(inp, (int(x), int(y)), (int(right), int(bottom)), (255,0,0), thickness=2)    
    

    print('Saving file name: ',name)
    result = Image.fromarray(inp)
    result.save('/data1/paritosh.pandey/CIGDataset/Results/'+name)
    print('Saved')
