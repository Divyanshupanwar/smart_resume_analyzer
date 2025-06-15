import pymysql
import streamlit as st
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME

def create_db_connection():
    """
    Create a connection to the database
    """
    try:
        connection = pymysql.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

def setup_database():
    """
    Setup database and tables if they don't exist
    """
    try:
        connection = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD)
        cursor = connection.cursor()
        
        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS SRA;")
        connection.select_db("sra")
        
        # Create user_data table with additional 'selected' column
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name varchar(100) NOT NULL,
            Email_ID VARCHAR(50) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field VARCHAR(25) NOT NULL,
            User_level VARCHAR(30) NOT NULL,
            Actual_skills VARCHAR(300) NOT NULL,
            Recommended_skills VARCHAR(300) NOT NULL,
            Recommended_courses VARCHAR(600) NOT NULL,
            total_score INT DEFAULT 0,
            selected BOOLEAN DEFAULT FALSE,
            PRIMARY KEY (ID)
        );
        """)
        
        connection.commit()
        return connection, cursor
    except Exception as e:
        st.error(f"Database Setup Error: {e}")
        return None, None

def insert_data(cursor, connection, name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, 
               skills, recommended_skills, courses, total_score=0, selected=False):
    """
    Insert data into the database
    """
    try:
        insert_sql = """
        INSERT INTO user_data (
            Name, Email_ID, resume_score, Timestamp, Page_no, 
            Predicted_Field, User_level, Actual_skills, 
            Recommended_skills, Recommended_courses, total_score, selected
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        rec_values = (
            name, email, str(res_score), timestamp, str(no_of_pages), 
            reco_field, cand_level, skills, recommended_skills, 
            courses, total_score, selected
        )
        
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Database Insert Error: {e}")
        return False

def get_all_resumes(cursor):
    """
    Get all resumes from the database
    """
    try:
        cursor.execute("""
        SELECT ID, Name, Email_ID, resume_score, Predicted_Field, 
               User_level, Actual_skills, total_score, selected 
        FROM user_data
        """)
        
        columns = ['id', 'name', 'email', 'resume_score', 'predicted_field', 
                  'experience_level', 'skills', 'total_score', 'selected']
        
        results = cursor.fetchall()
        resumes = []
        
        for row in results:
            resume_dict = dict(zip(columns, row))
            # Convert skills string to list
            if isinstance(resume_dict['skills'], str):
                resume_dict['skills'] = resume_dict['skills'].strip('[]').replace("'", "").split(', ')
            resumes.append(resume_dict)
            
        return resumes
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return []

def update_selected_status(cursor, connection, id_list, selected=True):
    """
    Update the selected status of resumes
    """
    try:
        # First reset all to not selected
        if selected:
            cursor.execute("UPDATE user_data SET selected = FALSE")
        
        # Then set selected for the chosen ones
        if id_list:
            placeholders = ', '.join(['%s'] * len(id_list))
            cursor.execute(f"UPDATE user_data SET selected = {selected} WHERE ID IN ({placeholders})", id_list)
            
        connection.commit()
        return True
    except Exception as e:
        st.error(f"Database Update Error: {e}")
        return False

def get_filtered_resumes(cursor, field=None, experience_level=None, min_score=0):
    """
    Get resumes filtered by field, experience level, and minimum score
    """
    query = """
    SELECT ID, Name, Email_ID, resume_score, Predicted_Field, 
           User_level, Actual_skills, total_score, selected 
    FROM user_data
    WHERE total_score >= %s
    """

    params = [min_score]

    if field:
        query += " AND Predicted_Field = %s"
        params.append(field)

    if experience_level:
        query += " AND User_level = %s"
        params.append(experience_level)

    try:
        cursor.execute(query, params)

        columns = ['id', 'name', 'email', 'resume_score', 'predicted_field', 
                   'experience_level', 'skills', 'total_score', 'selected']

        results = cursor.fetchall()
        resumes = []

        for row in results:
            resume_dict = dict(zip(columns, row))
            # Convert skills string to list
            if isinstance(resume_dict['skills'], str):
                resume_dict['skills'] = resume_dict['skills'].strip('[]').replace("'", "").split(', ')
            resumes.append(resume_dict)

        return resumes
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return []

def get_selected_resumes(cursor):
    """
    Get all selected resumes
    """
    try:
        cursor.execute("""
        SELECT ID, Name, Email_ID, Predicted_Field, User_level, total_score
        FROM user_data
        WHERE selected = TRUE
        """)
        
        columns = ['id', 'name', 'email', 'field', 'level', 'score']
        
        results = cursor.fetchall()
        selected_resumes = []
        
        for row in results:
            resume_dict = dict(zip(columns, row))
            selected_resumes.append(resume_dict)
            
        return selected_resumes
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return []
