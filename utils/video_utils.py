import yt_dlp
import random
from data.courses import resume_videos, interview_videos, ds_course, web_course, android_course, ios_course, uiux_course
import streamlit as st

def fetch_yt_video(link):
    """
    Fetch YouTube video title
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', None)
        
    return video_title

def course_recommender(course_list):
    """
    Recommend courses based on the provided course list
    """
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course
