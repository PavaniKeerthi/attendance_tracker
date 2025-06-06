import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io
from attendance_processor import AttendanceProcessor
from face_detector import FaceDetector

def main():
    st.set_page_config(page_title="Google Meet Attendance Tracker", layout="centered")
    st.title("Google Meet Attendance Tracker")
    st.write("Upload a screenshot of your Meet and a student CSV to track attendance automatically.")

    if 'attendance_results' not in st.session_state:
        st.session_state.attendance_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Meet Screenshot")
        uploaded_image = st.file_uploader(
            "Choose a screenshot",
            type=['png', 'jpg', 'jpeg'],
            help="Screenshot of Google Meet with participants visible"
        )
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Screenshot", use_container_width=True)

    with col2:
        st.subheader("Upload Student List")
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file with a 'Name' column"
        )
        df_students = None
        if uploaded_csv:
            try:
                df_students = pd.read_csv(uploaded_csv)
                st.write("Preview of Student List:")
                st.dataframe(df_students, use_container_width=True)

                if 'Name' not in df_students.columns:
                    st.error("CSV must contain a 'Name' column.")
                    df_students = None
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if uploaded_image and df_students is not None:
        if st.button("Process Attendance", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    image = Image.open(uploaded_image)
                    image_array = np.array(image)
                    image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if image_array.ndim == 3 else image_array

                    student_names = df_students['Name'].dropna().astype(str).tolist()

                    face_detector = FaceDetector()
                    attendance_processor = AttendanceProcessor()

                    faces_data = face_detector.detect_faces_and_video_status(image_cv)

                    attendance_results = attendance_processor.process_attendance(
                        student_names, faces_data, image_cv
                    )

                    st.session_state.attendance_results = attendance_results
                    st.session_state.processed_image = image_cv

                    st.success("Attendance processing completed!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Processing error: {e}")

    if st.session_state.attendance_results is not None:
        st.subheader("Attendance Results")
        results_df = pd.DataFrame(st.session_state.attendance_results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(results_df))
        col2.metric("Video On", len(results_df[results_df['Score'] == 100]))
        col3.metric("Video Off", len(results_df[results_df['Score'] == 50]))
        col4.metric("Absent", len(results_df[results_df['Score'] == 0]))

        st.dataframe(results_df, use_container_width=True)

        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Attendance Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="attendance_report.csv",
            mime="text/csv",
            type="primary"
        )

        if st.session_state.processed_image is not None:
            st.subheader("Processed Image with Face Detection")
            display_image = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
            st.image(display_image, caption="Detected Faces", use_container_width=True)

    with st.expander("How to use this app"):
        st.markdown("""
        1. Upload a Google Meet screenshot with visible participants.
        2. Upload a CSV file with a column named **'Name'** listing students.
        3. Click **Process Attendance** to analyze.
        4. Download the attendance report as a CSV.
        """)

if __name__ == "__main__":
    main()
