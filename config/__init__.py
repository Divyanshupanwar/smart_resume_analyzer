# Configuration constants can be defined here
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'div230111017@'
DB_NAME = 'sra'

# Paths
UPLOAD_FOLDER = './Uploaded_Resumes/'
LOGO_FOLDER = './static/Logo/'

# Ensure directories exist
import os
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGO_FOLDER, exist_ok=True)
