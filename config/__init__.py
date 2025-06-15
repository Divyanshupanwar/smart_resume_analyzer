# Configuration constants for the online MySQL database (freesqldatabase.com)

DB_HOST = 'sql12.freesqldatabase.com'
DB_USER = 'sql12784861'
DB_PASSWORD = '5qilebqMfs'  # 🔒 Replace with the real password you set
DB_NAME = 'sql12784861'


# Paths
UPLOAD_FOLDER = './Uploaded_Resumes/'
LOGO_FOLDER = './static/Logo/'

# Ensure directories exist
import os
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGO_FOLDER, exist_ok=True)
