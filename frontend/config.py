

# Put any configurations here that are common across all environments



class Config(object):
    """
    Common configurations
    """

    
"""
    Development configurations
    """

class DevelopmentConfig(Config):
 
    DEBUG = True
    SQLALCHEMY_ECHO = True
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 465
    MAIL_DEFAULT_SENDER = 'dhruvsai1151@gmail.com'
    MAIL_USERNAME = 'dhruvsai1151@gmail.com'
    MAIL_PASSWORD = 'Dedication@9635'
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    IMAGE_FOLDER = "images"

    # Celery configuration
    CELERY_BROKER_URL = 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

"""
    Production configurations
    """
    
class ProductionConfig(Config):
    

    DEBUG = False


app_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig
}