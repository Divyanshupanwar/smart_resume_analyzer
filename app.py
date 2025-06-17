import streamlit as st
from PIL import Image
import os
import time
import nltk

# 👇 Cloud-safe download path for nltk resources
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("stopwords", download_dir=nltk_data_path)

# Import modules from our refactored structure
from utils.file_utils import show_pdf
from utils.db_utils import setup_database, create_db_connection
from utils.video_utils import fetch_yt_video
from logic.scoring import ResumeRanker
from views.user_view import render_user_view
from views.admin_view import render_admin_view

def run():
    st.title("Enhanced Smart Resume Analyser")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    img = Image.open('./static/Logo/logo1.jpg') if os.path.exists('./static/Logo/logo1.jpg') else None
    if img:
        img = img.resize((250, 250))
        st.image(img)

    # Setup database
    connection, cursor = setup_database()
    if not connection or not cursor:
        st.error("Failed to connect to database. Please check your configuration.")
        return

    if choice == 'Normal User':
        render_user_view(connection, cursor)
    else:
        render_admin_view(connection, cursor)

if __name__ == "__main__":
    run()
