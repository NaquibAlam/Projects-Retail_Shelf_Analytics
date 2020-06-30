import os
import cv2
import glob
import numpy as np
from flask import Flask, redirect, request, url_for, current_app, flash, render_template,send_from_directory
from werkzeug.utils import secure_filename
import shelfShareCalc as ss
import stitch_images as si
from PIL import Image
import time
#import Image
app = Flask(__name__)
app.config['IMAGE_FOLDER'] = "images"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
RESULT_FOLDER = os.getcwd()
def testing():
    return "hello"


@app.route('/', methods=['GET', 'POST'])
def uploadsheet():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files.getlist('file[]')
        
        # if user does not select file, browser also
        # submit a empty part without filename
        if file[0].filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            uploads = os.path.join(current_app.root_path, app.config['IMAGE_FOLDER'])
            filepath = []

            for f in os.listdir('static/temp'):
                os.remove(os.path.join('static/temp',f))
            for f in os.listdir('static/images'):
                os.remove(os.path.join('static/images',f))
            for f in os.listdir(uploads):
                os.remove(os.path.join(uploads,f))
            
            for i in range(len(file)):
                filename = secure_filename(file[i].filename)
                filepath.append(os.path.join(uploads, filename))
                print(filename)
                file[i].save(os.path.join(uploads, filename))
            
            t0 = time.time()
            indices = si.main(filepath)
            print(time.time()-t0)
            urlImages = url_for('static',filename='')
            fileList = []
            
            i = 0
            while True:
                if os.path.isfile(os.path.join("static/temp","result_"+str(i)+".JPG")):
                    fileList.append("result_"+str(i)+".JPG")
                    i+=1
                    print("result_"+str(i)+".JPG")
                else:
                    break

            # for files in os.listdir('static/temp'):
            #     if files[0:7] == "resultC":
            #         print(files)
            #         fileList.append(files)
            
            # print("indices: ",indices)
            return render_template('inputDisplay.html',urlImages = urlImages+'temp/', 
                fileList = fileList, indices = indices, filepath = filepath)
    return render_template('firstPage.html')

@app.route('/inputDisplay')
def input_display():
    return render_template('inputDisplay.html')

@app.route('/output')
def result_disp():
    ss.main_fn("static/temp/");
    
    urlImages = url_for('static',filename='')
    length = len([f for f in os.listdir('static') if os.path.isfile('static/'+f)])/5
    dataTab = [None]*int(length)*3

    for file in os.listdir('static'):
        if (file[0:9] == "ProdCount"):
            print(file)
            f = open(os.path.join("static",file),"r")
            mylist = [line.rstrip('\n') for line in f]
            dataTab[3*(int(file[10:-4])-1)] = mylist
    
        if (file[0:9] == "ProdShare"):
            print(file)
            f = open(os.path.join("static",file),"r")
            mylist = [line.rstrip('\n') for line in f]
            dataTab[3*int(file[10:-4])-2] = mylist

        if (file[0:11] == "ProdVacancy"):
            print(file)
            f = open(os.path.join("static",file),"r")
            mylist = [line.rstrip('\n') for line in f]
            dataTab[3*int(file[12:-4])-1] = mylist
            # print([line for line in mylist[1].split(" ") if line!=""])
        
    return render_template('output.html', urlImages = urlImages, length = int(length), data = dataTab)

@app.route('/vacancy')
def vacancy_disp():
    vacancy_files = sorted(glob.glob('static/ProdVacancy*'))
    quantity_files = sorted(glob.glob('static/ProdCount*'))
    
    urlImages = url_for("static",filename="")
    all_vacancy_data = []
    all_quantity_total = []
    all_n_prods = []
    all_n_racks = []
    
    for filename_vacancy,filename_quantity in zip(vacancy_files, quantity_files):
        f_vacancy = open(filename_vacancy)
        f_quantity = open(filename_quantity)
        text_vacancy = f_vacancy.readlines()
        text_quantity = f_quantity.readlines()
        quantity_total = []
        vacancy_lines = []
    
        for line1,line2 in zip(text_vacancy[1:], text_quantity[1:]):
            vacancy_lines.append(line1.strip().split())
            quantity_total.append(line2.strip().split()[-2])
        all_vacancy_data.append(vacancy_lines)
        all_n_prods.append(len(vacancy_lines))
        all_n_racks.append(len(vacancy_lines[0])-1)
        all_quantity_total.append(quantity_total)

    skewData = [None]*int(len(vacancy_files))
    for file in os.listdir('static'):
        if (file[0:8] == "ProdSkew"):
            print(file)
            f = open(os.path.join("static",file),"r")
            mylist = [line.rstrip('\n') for line in f]
            skewData[int(file[9:-4])-1] = mylist
    
    return render_template('vacancy.html', vacancy_arr = all_vacancy_data, quantity_arr = all_quantity_total,
     n_prods_arr = all_n_prods, n_racks_arr = all_n_racks, urlImages = urlImages, skewData = skewData)

@app.route('/result')
def send_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5000,debug=True)

    
