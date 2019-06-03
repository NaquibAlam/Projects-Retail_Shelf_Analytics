# views.py



from flask import Flask, render_template, session, jsonify, request, flash, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from celery import Celery

# from flask_login import login_required, login_user, logout_user, current_user
# from flask_mail import Message
from itsdangerous import URLSafeTimedSerializer
# from sqlalchemy import text

# from app import db
# from app import mail
from app import app
from datetime import timedelta

import io
import time, random
import datetime
import pprint
import json
import uuid
import paramiko
import os, errno
import shlex,subprocess

# from PyPDF2 import PdfFileReader, PdfFileWriter
import pandas as pd
import numpy as np


# from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
# from bokeh.models.glyphs import VBar
# from bokeh.plotting import figure, output_file, show, save
# from bokeh.charts import Bar
# from bokeh.embed import components
# from bokeh.models.sources import ColumnDataSource
# from bokeh.models.ranges import DataRange1d
# from bokeh.palettes import Spectral11
# from bokeh.models.glyphs import MultiLine


from configparser import ConfigParser
import pickle


ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])

# Initialize Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0' 
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


def send_email(recipients_mailID, subject, html):

	msg = Message(subject = subject,
				  sender="dhruvsai1151@gmail.com",
				  recipients=[recipients_mailID],
				  html = html)

	mail.send(msg)

	return

@celery.task(bind=True)
def long_task(self):
    """Background task that runs a long function with progress reports."""
    verb = ['Starting up', 'Booting', 'Repairing', 'Loading', 'Checking']
    adjective = ['master', 'radiant', 'silent', 'harmonic', 'fast']
    noun = ['solar array', 'particle reshaper', 'cosmic ray', 'orbiter', 'bit']
    message = ''
    total = random.randint(10, 50)
    for i in range(total):
        if not message or random.random() < 0.25:
            message = '{0} {1} {2}...'.format(random.choice(verb),
                                              random.choice(adjective),
                                              random.choice(noun))
        self.update_state(state='PROGRESS',
                          meta={'current': i, 'total': total,
                                'status': message})
        time.sleep(1)
    return {'current': 100, 'total': 100, 'status': 'Task completed!', 'result': 42}
    
@app.route('/longtask', methods=['POST'])
def longtask():
    task = long_task.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    # print(task.info['status'])
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }

    return jsonify(response)


# Below functions are for opening fundamental pages

@app.route('/')
def index():
	return render_template('landing_page.html')

@app.route('/index')
def init():
	return render_template("index.html")

@app.route('/store')
@app.route('/setting')
def coming_soon():
	return render_template("coming_soon.html")

@app.route('/fileUpload')
def fileupload_html():
	return render_template("FileUpload.html")

@app.route('/blank')
def blank():
	return render_template("blank.html")

@app.route('/uploadFiles',methods=['POST'])
def uploadFiles():
	if request.method == 'POST':
		if 'file[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files.getlist('file[]')
		uuid = request.form['file'].replace("-", "")

		if file[0].filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file:
			# vSessionID = request.files['uuid']
			# print(vSessionID)
			uploads = os.path.join(current_app.root_path, app.config['IMAGE_FOLDER'])
			filepath = []
			# for f in os.listdir('static/temp'):
			# 	os.remove(os.path.join('static/temp',f))
			# for f in os.listdir(uploads):
			# 	os.remove(os.path.join(uploads,f))
			for i in range(len(file)):
				filename = secure_filename(file[i].filename)
				filepath.append(os.path.join(uploads, filename))
				print(filename)

				try:
					transport = paramiko.Transport(("172.20.2.26", 22))
					transport.connect(username = "paritosh.pandey", password = "#Apache35")
				except Exception as e:
					print(e)

				remote_path = os.path.join("/data1/paritosh.pandey", app.config['IMAGE_FOLDER'], str(uuid))
				# local_path = "/home/fractaluser/Predictive_Maintenance_files/" + vFileName
				sftp = paramiko.SFTPClient.from_transport(transport)
				try:
					sftp.chdir(remote_path)  # Test if remote_path exists
				except IOError:
					sftp.mkdir(remote_path)  # Create remote_path
					sftp.chdir(remote_path)
				file[i].save(os.path.join(uploads, filename))
				sftp.put(os.path.join(uploads, filename), os.path.join(remote_path, filename))
				os.remove(os.path.join(uploads, filename))
				# f.close()
				sftp.close()
				transport.close()


				# file[i].save(os.path.join(uploads, filename))
			return ('', 204)

@app.route('/dumpInput', methods=['POST'])
def dumpInputs():
	transport = paramiko.Transport(("172.20.2.26", 22))
	transport.connect(username = "paritosh.pandey", password = "#Apache35")
	remote_path = os.path.join("/data1/paritosh.pandey", app.config['IMAGE_FOLDER'])
				# local_path = "/home/fractaluser/Predictive_Maintenance_files/" + vFileName
	sftp = paramiko.SFTPClient.from_transport(transport)
	try:
		sftp.chdir(remote_path)  # Test if remote_path exists
	except IOError:
		stderr('Image folder absent')

	# for file in sftp.listdir(remote_path):
	# 	print(file)
	removalPath = os.path.join(remote_path, str(request.form['uuid']).replace("-", ""))
	print(removalPath)
	try:
		for file in sftp.listdir(removalPath):
			print(file)
			sftp.remove(os.path.join(removalPath, file))

		sftp.rmdir(removalPath)
	except Exception as e:
		stderr(e)
	
	sftp.close()
	transport.close()

	return ('', 204)

@app.route('/stitchAction', methods=['POST'])
def stitchAction():
	try:
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect('172.20.2.26',username='paritosh.pandey',password='#Apache35')
		stdin, stdout, stderr = ssh.exec_command('nohup python3 /data1/paritosh.pandey/shelfExec.py '+str(request.form['uuid']).replace("-", "") + ' 0 >/dev/null 2>&1 &')
		# stdout.channel.recv_exit_status()
		while True:
		  line = stdout.readline()
		  if line != '':
		    #the real code does filtering here
		    print(line)
		  else:
		    break
		while True:
		  line = stderr.readline()
		  if line != '':
		    #the real code does filtering here
		    print(line)
		  else:
		    break
	
		ssh.close()
		return ('', 204)

	except:
		return ('', 404)

@app.route('/calcAction', methods=['POST'])
def calcAction():
	try:
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect('172.20.2.26',username='paritosh.pandey',password='#Apache35')
		
		stdin, stdout, stderr = ssh.exec_command('cd /data1/paritosh.pandey; python3 shelfShareCalc.py '+str(request.form['uuid']).replace("-", ""))
		# stdout.channel.recv_exit_status()
		print(stdout.read())
		ssh.close()
		
		return ('', 204)

	except:
		return ('', 404)
	
@app.route('/getFolder', methods=['GET', 'POST'])
def readFolders():
	try:
		# print("Enter")
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect('172.20.2.26', username='paritosh.pandey', password="#Apache35")
		
		print("Start")
		stdin, stdout, stderr = ssh.exec_command('nohup python3 /data1/paritosh.pandey/shelfExec.py result /data1/paritosh.pandey/static/results/')
		folders = []
		print(stderr.read())
		for line in stdout:
			folders.append(line.rstrip('\n'))

		data = []
		for folder in folders:
			print(folder)
			stdin, stdout, stderr = ssh.exec_command('nohup python3 /data1/paritosh.pandey/shelfExec.py result table /data1/paritosh.pandey/static/results/'+folder)
			text = []
			for f in stdout:
				text.append(f.rstrip('\n'))
			data.append(text)

		ssh.close()

		# data = [['11-05-2018 13:27:10','c93db604e36e416795dbd35ad1264100','ABC','3']]#,['1','a','ABC','2'],['1','b','ABC','5']]

		return jsonify({'data': data})#stdout.read()})

	except:
		return ('', 404)

@app.route('/getData', methods=['GET', 'POST'])
def getData():
	try:
		ssh = paramiko.SSHClient()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		ssh.connect('172.20.2.26', username='paritosh.pandey', password="#Apache35")
		
		stdin, stdout, stderr = ssh.exec_command('nohup python3 /data1/paritosh.pandey/dataExtraction.py /data1/paritosh.pandey/static/results/'+str(request.form['uuid']))
		ln = int(request.form['len'])
		print(request.form['len'], ln)
		ProdSkew = [None]*ln; ProdQuant = [None]*ln; ProdShare = [None]*ln; ProdVac = [None]*ln; ProdHorz = [None]*ln; Facings = [None]*ln;
		fileText = []
		i = 0
		for line in stdout:
			word = line.rstrip("\n")
			# print(int(word))
			if i == 0:
				# ProdSkew = [None]*int(word)
				# ProdShare = [None]*int(word)
				# ProdQuant = [None]*int(word)
				# ProdVac = [None]*int(word)
				# Facings = [None]*int(word)
				i += 1
				continue
		
			if word[:4] != 'Prod':
				fileText.append(word)
			else:
				pos = word.index("_")
				index = int(word[pos+1:-4])
				if word[:6] == "ProdSk":
					ProdSkew[index-1] = fileText
				elif word[:6] == "ProdSh":
					ProdShare[index-1] = fileText
				elif word[:6] == "ProdCo":
					ProdQuant[index-1] = fileText
					count = 0
					for state in fileText:
						forF = list(filter(bool, state.split(' ')))
						if forF[-2] != "Total":
							count += float(forF[-2])
					Facings[index-1] = count
				elif word[:6] == "ProdVa":
					ProdVac[index-1] = fileText
				elif word[:6] == "ProdHo":
					ProdHorz[index-1] = fileText
				fileText = []
				
		ssh.close()

		# print("coming")
		# ProdSkew = [None]*3; ProdQuant = [None]*3; ProdShare = [None]*3; ProdVac = [None]*3; Facings = [None]*3;
		# for file in os.listdir('app/static/images/results'):
		# 	print(file)
		# 	if (file[-4:] == ".jpg") or (file == "timeStamp.txt"):
		# 		continue
				
		# 	f = open(os.path.join('app/static/images/results', file), "r")
		# 	fileText = []

		# 	pos = file.index("_")
		# 	index = int(file[pos+1:-4])
		# 	count = 0;
			
		# 	if (file[0:9] == "ProdCount"):
		# 		for line in f:
		# 			fileText.append(line.rstrip('\n'))
		# 			forF = list(filter(bool, fileText[-1].split(' ')))
		# 			# print(forF, forF[-2])
		# 			if forF[-2] != "Total":
		# 				# print('inside')
		# 				count += float(forF[-2])

		# 		ProdQuant[index-1] = fileText
		# 		Facings[index-1] = count
		# 		print(count)

		# 	if (file[0:9] == "ProdShare"):
		# 		for line in f:
		# 			fileText.append(line.rstrip('\n'))
		# 		ProdShare[index-1] = fileText

		# 	if (file[0:8] == "ProdSkew"):
		# 		for line in f:
		# 			fileText.append(line.rstrip('\n'))
		# 		ProdSkew[index-1] = fileText

		# 	if (file[0:11] == "ProdVacancy"):
		# 		for line in f:
		# 			fileText.append(line.rstrip('\n'))
		# 		ProdVac[index-1] = fileText
    
		return jsonify({'ProdSkew': ProdSkew, 'ProdQuant' : ProdQuant, 'ProdShare' : ProdShare, 'ProdVac' : ProdVac, 'ProdHorz' : ProdHorz, "Facings" : Facings})
	except:
		return(jsonify({"Status": "Failure"}))

@app.route('/downloadImages', methods=['POST'])
def downloadImages():

	transport = paramiko.Transport(("172.20.2.26", 22))
	transport.connect(username = "paritosh.pandey", password = "#Apache35")
	remote_path = "/data1/paritosh.pandey"
	
	sftp = paramiko.SFTPClient.from_transport(transport)
	try:
		sftp.chdir(remote_path)  # Test if remote_path exists
	except IOError:
		stderr('Path error')

	folderName = request.form['uuid']
	noOfInput = []

	# inputPath = os.path.join(remote_path, 'images')
	uploadsI = os.path.join(current_app.root_path, 'static/images/input')
	for folder in os.listdir(uploadsI):
		if not os.path.isfile(os.path.join(uploadsI, folder)):
			for file in os.listdir(os.path.join(uploadsI, folder)):
				os.remove(os.path.join(os.path.join(uploadsI, folder), file))
		else:
			os.remove(os.path.join(uploadsI, folder))
	# for file in sftp.listdir(os.path.join(inputPath, folderName)):
	# 	sftp.get(os.path.join(os.path.join(inputPath, folderName), file), os.path.join(uploads, file))

	stitchPath = os.path.join(remote_path, 'static/temp')
	uploads = os.path.join(current_app.root_path, 'static/images/stitch')
	for file in os.listdir(uploads):
		os.remove(os.path.join(uploads, file))
	for folder in sftp.listdir(os.path.join(stitchPath, folderName)):
		print(folder)
		if folder[-4:] == ".jpg":
			print("Copy Results")
			sftp.get(os.path.join(os.path.join(stitchPath, folderName), folder), os.path.join(uploads, folder))
		else:
			try:
				os.mkdir(os.path.join(uploadsI, folder))
				print("Made Dir")
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
				pass
			print(os.path.join(os.path.join(stitchPath, folderName), folder))
			count = 0
			for file in sftp.listdir(os.path.join(os.path.join(stitchPath, folderName), folder)):
				print(os.path.join(os.path.join(uploadsI, folder), file), os.path.join(os.path.join(uploadsI, folder), str(count)+".jpg"))
				sftp.get(os.path.join(os.path.join(os.path.join(stitchPath, folderName), folder), file), os.path.join(os.path.join(uploadsI, folder), file))
				os.rename(os.path.join(os.path.join(uploadsI, folder), file), os.path.join(os.path.join(uploadsI, folder), "input_"+str(count)+".jpg"))
				count += 1
			noOfInput.append(count)		

	resultsPath = os.path.join(remote_path, 'static/results')
	uploads = os.path.join(current_app.root_path, 'static/images/results')
	for file in os.listdir(uploads):
		os.remove(os.path.join(uploads, file))
	for file in sftp.listdir(os.path.join(resultsPath, folderName)):
		# if file[-4:] == '.txt':
		# 	continue
		sftp.get(os.path.join(os.path.join(resultsPath, folderName), file), os.path.join(uploads, file))
	
	sftp.close()
	transport.close()

	# noOfInput = [3,3,3]

	return jsonify({"noOfInput": noOfInput})

@app.route('/results', methods=['POST', 'GET'])
def results():
	return render_template("results.html")