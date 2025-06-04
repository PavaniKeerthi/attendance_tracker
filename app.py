import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io
from attendance_processor import AttendanceProcessor
from face_detector import FaceDetector

def main():
    st.title("Google Meet Attendance Tracker")
    st.write("Upload a Google Meet screenshot and student list to automatically track attendance")
    
    # Initialize session state
    if 'attendance_results' not in st.session_state:
        st.session_state.attendance_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Meet Screenshot")
        uploaded_image = st.file_uploader(
            "Choose a screenshot file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a screenshot of your Google Meet session"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Screenshot", use_container_width=True)
    
    with col2:
        st.subheader("Upload Student List")
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with student names (header: 'Name')"
        )
        
        df_students = None
        if uploaded_csv is not None:
            try:
                df_students = pd.read_csv(uploaded_csv)
                st.write("Student List Preview:")
                st.dataframe(df_students, use_container_width=True)
                
                if 'Name' not in df_students.columns:
                    st.error("CSV file must contain a 'Name' column")
                    return
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
    
    # Process attendance
    if uploaded_image is not None and df_students is not None:
        if st.button("Process Attendance", type="primary"):
            with st.spinner("Processing attendance... This may take a moment."):
                try:
                    # Convert uploaded image to OpenCV format
                    image = Image.open(uploaded_image)
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_cv = image_array
                    
                    student_names = df_students['Name'].tolist()
                    
                    # Initialize processors
                    face_detector = FaceDetector()
                    attendance_processor = AttendanceProcessor()
                    
                    # Detect faces and analyze video status
                    faces_data = face_detector.detect_faces_and_video_status(image_cv)
                    
                    # Process attendance
                    attendance_results = attendance_processor.process_attendance(
                        student_names, faces_data, image_cv
                    )
                    
                    # Store results
                    st.session_state.attendance_results = attendance_results
                    st.session_state.processed_image = image_cv
                    
                    st.success("Attendance processing completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing attendance: {str(e)}")
    
    # Display results
    if st.session_state.attendance_results is not None:
        st.subheader("Attendance Results")
        
        results_df = pd.DataFrame(st.session_state.attendance_results)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(results_df))
        with col2:
            st.metric("Video On", (results_df['Score'] == 100).sum())
        with col3:
            st.metric("Video Off", (results_df['Score'] == 50).sum())
        with col4:
            st.metric("Absent", (results_df['Score'] == 0).sum())
        
        # Detailed table
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Attendance Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="attendance_report.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Show processed image
        if st.session_state.processed_image is not None:
            st.subheader("Face Detection Results")
            display_image = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption="Processed Image with Face Detection", use_container_width=True)
    
    # Instructions
    with st.expander("How to use this application"):
        st.write("""
        **Steps:**
        1. Take a screenshot of your Google Meet session showing all participants
        2. Prepare a CSV file with student names (column header must be 'Name')
        3. Upload both files using the upload buttons above
        4. Click 'Process Attendance' to analyze the screenshot
        5. Download the attendance report as a CSV file
        
        **Scoring System:**
        - Video On (face visible in video feed): 100 points
        - Video Off (profile picture shown): 50 points
        - Absent (not detected in screenshot): 0 points
        
        **Tips for better results:**
        - Use high-quality screenshots
        - Ensure student names in CSV match their Google Meet display names
        - Make sure all participants are visible in the screenshot
        """)

if __name__ == "__main__":
    main()
