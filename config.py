class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres:vocare123@localhost:5432/vocare"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "vocarecare"
    JWT_SECRET_KEY = "vocarecare_jwt"
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
