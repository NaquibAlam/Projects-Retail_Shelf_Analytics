import os
import cv
import numpy as np
from PIL import Image

image_path = '/data1/paritosh.pandey/CIGDataset/ShelfImages'
annotations_path = '/data1/paritosh.pandey/CIGDataset/BBOX'
for filename in os.listdir(path):
   img = Image.open(path + '/' + filename)
   ing = np.array(img)
   ing = ing[:, :, ::-1].copy()
   

