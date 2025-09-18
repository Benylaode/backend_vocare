from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flasgger import Swagger

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    swagger = Swagger(app, template_file='swagger.yaml')


    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    

    from .routes.main import main_bp
    from .routes.user import user_bp
    from .routes.auth import auth_bp
    from .routes.assesments import assesment_bp
    from .routes.pdf import document_bp as pdf
    from .routes.laporan import laporan_bp as laporan
    from .routes.cppt import cppt_bp as cppt
    from .routes.patient import patient_bp as patient
    app.register_blueprint(patient, url_prefix='/patients')
    app.register_blueprint(cppt, url_prefix='/cppt')
    app.register_blueprint(laporan, url_prefix='/laporan')
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(main_bp)
    app.register_blueprint(user_bp, url_prefix='/user')
    app.register_blueprint(assesment_bp, url_prefix='/assesments')
    app.register_blueprint(pdf, url_prefix='/pdf')

    return app
