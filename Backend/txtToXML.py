import xml.etree.ElementTree as ET
import itertools as it
import xml.dom.minidom as minidom
import os
from PIL import Image as img
#root = ET.Element('annotations')

with open('/home/fractaluser/Downloads/annotation.txt') as f:
    lines = f.read().splitlines()

filepath = "/home/fractaluser/Documents/BBOX/"
#add first subelement
#celldata = ET.SubElement(root, 'filename')

#import itertools as it
#for every line in input file
#group consecutive dedup to one 
for line in it.groupby(lines):
    line=line[0]
    #if its a break of subelements  - that is an empty space
    if not line:
        print()
        #add the next subelement and get it as celldata
        #celldata = ET.SubElement(root, 'filedata')
    else:
        #otherwise, split with : to get the tag name
        tag = line.split(" ")
    
        root = ET.Element('annotations')
        folder = ET.SubElement(root, 'folder')
        filename = ET.SubElement(root, 'filename')
        filename.text = tag[0]
        #objects = ET.SubElement(root,'NoOfObjects')
        #objects.text = tag[1]
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size,'width')
        height = ET.SubElement(size,'height')
        depth = ET.SubElement(size,'depth')
        ET.SubElement(root,'segmented').text = str(0)
        
        if os.path.exists("/home/fractaluser/Documents/ShelfImages/Train/"+tag[0]):
            folder.text = "Train"
            path = os.path.join("/home/fractaluser/Documents/ShelfImages/Train/",tag[0])
        else:
            folder.text = "Val"
            path = os.path.join("/home/fractaluser/Documents/ShelfImages/Val/",tag[0])
            
        im = img.open(path)
        size = im.size
        height.text = str(size[0])
        width.text = str(size[1])
        depth.text = str(3)
        
        for bb in range(int(tag[1])):
            obj = ET.SubElement(root,'object')
            name = ET.SubElement(obj,'name')
            if int(tag[6+5*bb])+1 == 11:
                name.text = str(10)
            else:
                name.text = str(int(tag[6+5*bb])+1)
            
            pose = ET.SubElement(obj,'pose')
            pose.text = 'Frontal'
            truncated = ET.SubElement(obj,'truncated')
            truncated.text = str(0)
            difficult = ET.SubElement(obj,'difficult')
            difficult.text = str(0)
            
            bbox = ET.SubElement(obj,'bndbox')
            x = ET.SubElement(bbox,'xmin')
            x.text = tag[2+5*bb]
            y = ET.SubElement(bbox,'ymin')
            y.text = tag[3+5*bb]
            w = ET.SubElement(bbox,'xmax')
            w.text = str(int(tag[2+5*bb]) + int(tag[4+5*bb]))
            h = ET.SubElement(bbox,'ymax')
            h.text = str(int(tag[3+5*bb]) + int(tag[5+5*bb]))
            
        formatedXML = minidom.parseString(
                    ET.tostring(
                            root)).toprettyxml(indent=" ",encoding='utf-8').strip()
            
        filepath = "/home/fractaluser/Documents/BBOX/"
        with open(filepath+tag[0][:-4]+".xml","wb+") as f:
            f.write(formatedXML)


        #format tag name
        #el=ET.SubElement(celldata,tag[0].replace(" ",""))
        #tag=' '.join(tag[1:]).strip()

        #get file name from file path
        #if 'File Name' in line:
            #tag = line.split("\\")[-1].strip()
        #elif 'File Size' in line:
            #splist =  filter(None,line.split(" "))
            #tag = splist[splist.index('Low:')+1]
            #splist[splist.index('High:')+1]
        #el.text = tag

#prettify xml
#import xml.dom.minidom as minidom
#formatedXML = minidom.parseString(
                          #ET.tostring(
                                      #root)).toprettyxml(indent=" ",encoding='utf-8').strip()
# Display for debugging
#print formatedXML

#write the formatedXML to file.
#with open("Performance.xml","w+") as f:
    #f.write(formatedXML)

#filepath = "/home/fractaluser/Documents/ShelfImages/*.JPG"
#import glob as gb
#filelist = gb.glob(filepath)
#
#from random import shuffle
#filearange = [i for i in range(len(filelist))]
#shuffle(filearange)
#
#from shutil import copyfile
#for i in range(len(filelist)):
#    if i < 300:
#        copyfile(filelist[filearange[i]],'/home/fractaluser/Documents/ShelfImages/Train/'+filelist[filearange[i]][-18:])
#    else:
#        copyfile(filelist[filearange[i]],'/home/fractaluser/Documents/ShelfImages/Val/'+filelist[filearange[i]][-18:])