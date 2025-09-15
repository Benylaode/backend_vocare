class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://vocare:vocare@localhost:5432/vocare"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "vocarecare" 
    JWT_SECRET_KEY= "vocarecare_jwt"  
