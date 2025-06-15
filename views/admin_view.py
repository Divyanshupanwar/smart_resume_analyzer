import streamlit as st
import pandas as pd
import plotly.express as px
from utils.db_utils import get_all_resumes, update_selected_status, get_filtered_resumes, get_selected_resumes
from utils.file_utils import get_table_download_link
from logic.scoring import ResumeRanker

def render_admin_view(connection, cursor):
    """
    Render the admin view for resume management and analytics
    """
    # Initialize session state variables
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False
    if 'filtered_resumes' not in st.session_state:
        st.session_state.filtered_resumes = []
    if 'job_search_results' not in st.session_state:
        st.session_state.job_search_results = []
    
    # If not logged in, show login form
    if not st.session_state.admin_logged_in:
        st.success('Welcome to Admin Side')
        
        with st.form("admin_login"):
            ad_user = st.text_input("Username")
            ad_password = st.text_input("Password", type='password')
            login_button = st.form_submit_button('Login')
            
            if login_button:
                if ad_user == 'machine_learning_hub' and ad_password == 'mlhub123':
                    st.session_state.admin_logged_in = True
                    st.success("Welcome Admin")
                    st.rerun()
                else:
                    st.error("Wrong ID & Password Provided")
    
    # If logged in, show admin dashboard
    else:
        st.success("Welcome Admin")
        
        # Add logout button
        if st.sidebar.button("Logout"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        # Admin tabs for different functionalities
        admin_tabs = st.tabs(["User Data", "Best Resumes Selection", "Analytics"])
        
        with admin_tabs[0]:
            # Display all user data
            try:
                # First, check what columns actually exist in the database
                cursor.execute("DESCRIBE user_data")
                db_columns = [column[0] for column in cursor.fetchall()]
                
                # Execute the query with explicit column selection
                cursor.execute('''
                SELECT ID, Name, Email_ID, resume_score, Timestamp, Page_no,
                       Predicted_Field, User_level, Actual_skills, Recommended_skills,
                       Recommended_courses
                FROM user_data
                ''')
                
                data = cursor.fetchall()
                st.header("**User's👨‍💻 Data**")
                
                # Create DataFrame with only the columns we have
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                               'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                               'Recommended Course'])
                
                # Check if total_score and selected columns exist in the database
                if 'total_score' in db_columns and 'selected' in db_columns:
                    # If they exist, add them to the query
                    cursor.execute('''
                    SELECT total_score, selected FROM user_data
                    ''')
                    additional_data = cursor.fetchall()
                    
                    # Add these columns to the DataFrame
                    if len(additional_data) == len(data):
                        df['Total Score'] = [row[0] for row in additional_data]
                        df['Selected'] = [row[1] for row in additional_data]
                
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error displaying user data: {e}")
                # Fallback to simpler query if the above fails
                cursor.execute('''SELECT * FROM user_data''')
                data = cursor.fetchall()
                # Get column names directly from cursor description
                columns = [column[0] for column in cursor.description]
                df = pd.DataFrame(data, columns=columns)
                st.dataframe(df)
        
        with admin_tabs[1]:
            # Best resumes selection using DAA
            st.header("**Best Resumes Selection**")
            
            # Add filtering options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Filter by field
                field_filter = st.selectbox(
                    "Filter by Field",
                    ["All Fields", "Data Science", "Web Development", "Android Development", 
                     "IOS Development", "UI-UX Development"]
                )
            
            with col2:
                # Filter by experience level
                level_filter = st.selectbox(
                    "Filter by Experience Level",
                    ["All Levels", "Fresher", "Intermediate", "Experienced"]
                )
            
            with col3:
                # Minimum score
                min_score = st.slider("Minimum Score", 0, 500, 100)
            
            # Number of resumes to select
            num_resumes = st.slider("Number of Resumes to Select", 1, 20, 5)
            
            # Filter and sort button - with a unique key
            if st.button("Find Best Resumes", key="find_best_resumes"):
                # Convert "All Fields" to None for the filter function
                field = None if field_filter == "All Fields" else field_filter
                level = None if level_filter == "All Levels" else level_filter
            
                # Fetch filtered resumes
                filtered_resumes = get_filtered_resumes(cursor, field, level, min_score)
                
                if filtered_resumes:
                    # Automatically select the best algorithm based on data size
                    if len(filtered_resumes) > 100:
                        algorithm = 'heap_sort'  # Best for large datasets
                    elif len(filtered_resumes) > 20:
                        algorithm = 'quick_sort'  # Good for medium datasets
                    else:
                        algorithm = 'merge_sort'  # Stable for small datasets
                    
                    st.write(f"Found {len(filtered_resumes)} matching resumes. Using optimal sorting algorithm.")
                    sorted_resumes = ResumeRanker.select_best_resumes(filtered_resumes, algorithm=algorithm)
                    st.session_state.filtered_resumes = sorted_resumes[:num_resumes]
                    
                    # FIXED: Automatically select these resumes in the database
                    selected_ids = [r['id'] for r in st.session_state.filtered_resumes]
                    if update_selected_status(cursor, connection, selected_ids):
                        st.success(f"Successfully selected {len(selected_ids)} best resumes.")
                    
                    # Display sorted resumes
                    st.subheader(f"Top {num_resumes} Resumes (Automatically Selected)")
                    sorted_df = pd.DataFrame([
                        {
                            'ID': r['id'], 
                            'Name': r['name'], 
                            'Email': r['email'], 
                            'Field': r['predicted_field'],
                            'Level': r['experience_level'],
                            'Score': r['total_score']
                        } for r in st.session_state.filtered_resumes
                    ])
                    st.dataframe(sorted_df)
                    
                    # ADDED: Download button for selected resumes
                    st.markdown(get_table_download_link(sorted_df, 'Selected_Resumes.csv', 'Download Selected Resumes'), unsafe_allow_html=True)
                else:
                    st.warning("No resumes found matching the criteria.")
                    st.session_state.filtered_resumes = []
            
            # FIX: Add a section to view currently selected resumes
            st.subheader("Currently Selected Resumes")
            selected_resumes = get_selected_resumes(cursor)
            if selected_resumes:
                selected_df = pd.DataFrame(selected_resumes)
                st.dataframe(selected_df)
                
                # ADDED: Download button for currently selected resumes
                st.markdown(get_table_download_link(selected_df, 'Selected_Resumes.csv', 'Download Selected Resumes'), unsafe_allow_html=True)
                
                # Add option to clear selection with a unique key
                if st.button("Clear All Selections", key="clear_selections"):
                    if update_selected_status(cursor, connection, [], False):
                        st.success("All selections cleared.")
                        st.rerun()
            else:
                st.info("No resumes are currently selected.")
            
            # Job position search
            st.subheader("Quick Job Position Search")
            
            # Use a form to prevent rerun issues
            with st.form("job_search_form"):
                job_position = st.text_input("Enter Job Position (e.g., 'Data Scientist', 'Web Developer')")
                job_num_resumes = st.slider("Number of Candidates to Find", 1, 20, 5, key="job_num_slider")
                search_button = st.form_submit_button("Find Top Candidates")
            
            if search_button:
                if job_position:
                    field_mapping = {
                        'data scientist': 'Data Science',
                        'data analyst': 'Data Science',
                        'machine learning': 'Data Science',
                        'web developer': 'Web Development',
                        'frontend developer': 'Web Development',
                        'backend developer': 'Web Development',
                        'full stack developer': 'Web Development',
                        'android developer': 'Android Development',
                        'ios developer': 'IOS Development',
                        'ui designer': 'UI-UX Development',
                        'ux designer': 'UI-UX Development'
                    }
                    job_lower = job_position.lower()
                    matched_field = None
                    for key, value in field_mapping.items():
                        if key in job_lower:
                            matched_field = value
                            break
                    if matched_field:
                        field_resumes = get_filtered_resumes(cursor, matched_field)
                        if field_resumes:
                            # Automatically select algorithm based on data size
                            if len(field_resumes) > 100:
                                algorithm = 'heap_sort'
                            elif len(field_resumes) > 20:
                                algorithm = 'quick_sort'
                            else:
                                algorithm = 'merge_sort'
                                
                            sorted_resumes = ResumeRanker.select_best_resumes(field_resumes, algorithm=algorithm)
                            st.session_state.job_search_results = sorted_resumes[:job_num_resumes]
                            
                            # FIXED: Automatically select these resumes in the database
                            selected_ids = [r['id'] for r in st.session_state.job_search_results]
                            if update_selected_status(cursor, connection, selected_ids):
                                st.success(f"Successfully selected {len(selected_ids)} top candidates for {job_position}.")
                            
                            st.subheader(f"Top {job_num_resumes} Candidates for {job_position} (Automatically Selected)")
                            top_df = pd.DataFrame([
                                {
                                    'ID': r['id'], 
                                    'Name': r['name'], 
                                    'Email': r['email'], 
                                    'Field': r['predicted_field'],
                                    'Level': r['experience_level'],
                                    'Score': r['total_score']
                                } for r in st.session_state.job_search_results
                            ])
                            st.dataframe(top_df)
                            
                            # ADDED: Download button for selected candidates
                            st.markdown(get_table_download_link(top_df, f'Selected_{job_position}_Candidates.csv', 'Download Selected Candidates'), unsafe_allow_html=True)
                        else:
                            st.warning(f"No resumes found for {job_position} position.")
                            st.session_state.job_search_results = []
                    else:
                        st.warning(f"Could not map '{job_position}' to any known job fields. Please try another search term.")
                        st.session_state.job_search_results = []
                else:
                    st.warning("Please enter a job position to search for.")
        
        with admin_tabs[2]:
            # Analytics
            st.header("**Analytics Dashboard**")
            
            try:
                # Load data for analytics
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                
                if not plot_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart for predicted field recommendations
                        if 'Predicted_Field' in plot_data.columns and not plot_data.Predicted_Field.empty:
                            labels = plot_data.Predicted_Field.unique()
                            values = plot_data.Predicted_Field.value_counts()
                            st.subheader("**Predicted Field Distribution**")
                            fig = px.pie(plot_data, values=values, names=labels, 
                                        title='Predicted Field according to Skills')
                            st.plotly_chart(fig)
                        else:
                            st.info("No data available for Predicted Field Distribution")
                    
                    with col2:
                        # Pie chart for User's Experience Level
                        if 'User_level' in plot_data.columns and len(plot_data.User_level.unique()) > 0:
                            labels = plot_data.User_level.unique()
                            values = plot_data.User_level.value_counts()
                            st.subheader("**Experience Level Distribution**")
                            fig = px.pie(plot_data, values=values, names=labels, 
                                        title="Users Experience Level")
                            st.plotly_chart(fig)
                        else:
                            st.info("No data available for Experience Level Distribution")
                    
                    # Score distribution histogram
                    if 'total_score' in plot_data.columns and not plot_data.total_score.empty:
                        st.subheader("**Score Distribution**")
                        fig = px.histogram(plot_data, x='total_score', 
                                          title='Distribution of Total Scores',
                                          nbins=10)
                        st.plotly_chart(fig)
                    else:
                        st.info("No data available for Score Distribution")
                    
                    # Selection analytics
                    if 'selected' in plot_data.columns:
                        selected_count = plot_data['selected'].sum()
                        total_count = len(plot_data)
                        
                        st.subheader("**Selection Statistics**")
                        st.write(f"Total Resumes: {total_count}")
                        st.write(f"Selected Resumes: {selected_count}")
                        if total_count > 0:
                            st.write(f"Selection Rate: {selected_count/total_count:.2%}")
                        
                        # Compare selected vs non-selected
                        if selected_count > 0 and (total_count - selected_count) > 0 and 'total_score' in plot_data.columns:
                            st.subheader("**Selected vs Non-Selected Comparison**")
                            avg_score_selected = plot_data[plot_data['selected'] == True]['total_score'].mean()
                            avg_score_nonselected = plot_data[plot_data['selected'] == False]['total_score'].mean()
                            
                            comparison_data = pd.DataFrame({
                                'Category': ['Selected', 'Non-Selected'],
                                'Average Score': [avg_score_selected, avg_score_nonselected]
                            })
                            
                            fig = px.bar(comparison_data, x='Category', y='Average Score',
                                        title='Average Score Comparison')
                            st.plotly_chart(fig)
                            
                            # ADDED: Download selected resumes from analytics tab
                            selected_data = plot_data[plot_data['selected'] == True]
                            if not selected_data.empty:
                                st.subheader("**Download Selected Resumes**")
                                st.markdown(get_table_download_link(selected_data, 'Selected_Resumes_Analytics.csv', 'Download Selected Resumes'), unsafe_allow_html=True)
                else:
                    st.info("No data available for analytics. Please add some resumes first.")
            except Exception as e:
                st.error(f"Error loading analytics: {e}")
                st.info("No data available for analytics. Please check database connection.")
