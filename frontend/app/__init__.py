from flask import Flask
from config import app_config

app = Flask(__name__, instance_relative_config=True)   

def create_app(config_name):
	
	# Load the config 
	print(config_name)
	app.config.from_object(app_config[config_name])
	app.config.from_pyfile('config.py')
	
	# Load the views
	from app import views

	return app
