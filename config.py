class Config:
    SQLALCHEMY_DATABASE_URI = "postgresql://postgres.tsrhbyhpsbwrgmauvzur:Vocare123%21%40%23@aws-1-us-east-2.pooler.supabase.com:6543/postgres"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = "vocarecare" 
    JWT_SECRET_KEY= "vocarecare_jwt"  
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# postgresql://postgres.tsrhbyhpsbwrgmauvzur:Vocare123%21%40%23@aws-1-us-east-2.pooler.supabase.com:6543/postgres
