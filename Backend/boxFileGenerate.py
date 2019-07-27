import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle as pk
import os

graph_def = None

def __init__(path):
    # Singleton declaration of the object
    # Read the graph.
    with tf.gfile.FastGFile(path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print('init finished...')
    return graph_def

def _img_load_process(path):
    img = Image.open(path)
    ing = np.array(img)
    ing = ing[:, :, ::-1].copy()
    ing = cv.resize(ing,(600,600))
    return ing


def _execution(ing,graph_def):
    print("Starting evaluation of image...")
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                sess.graph.get_tensor_by_name('detection_scores:0'),
                sess.graph.get_tensor_by_name('detection_boxes:0'),
                sess.graph.get_tensor_by_name('detection_classes:0')],
               feed_dict={'image_tensor:0': ing.reshape(1, ing.shape[0], ing.shape[1], 3)})
    print("Bounding boxes calculated")
    return out

def main():
    print('Main starts...')
    graph_path = '/data1/paritosh.pandey/model/frozen_inference_graph.pb'
    obj_img = __init__(graph_path)
    
    for filename in os.listdir('/data1/paritosh.pandey/test/'):
        image_under_test = _img_load_process('/data1/paritosh.pandey/test/'+filename)
        output = _execution(image_under_test,obj_img)
        with open('/data1/paritosh.pandey/'+filename+'.pkl','wb') as f:
            pk.dump(output,f)
    
    return


print('Execution')
main()
print('All is revealed.')
