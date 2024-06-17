import streamlit as st
from Home import face_rec
import cv2
import numpy as np

st.subheader('Registration Form')

# Initialize registration form
registeration_form = face_rec.RegisterationForm()

# Collect person name and email
person_name = st.text_input(label='Name', placeholder='First & Last Name')
person_email = st.text_input(label='email')

# Input source selection
source = st.radio("Choose the input source:", ("Webcam", "RTSP Stream"))

# RTSP URL
rtsp_url = 0        # Camera IP Address

# Placeholder for the video frame
frame_placeholder = st.empty()

# Start RTSP stream
def show_rtsp_stream(url):
    cap = cv2.VideoCapture(url)
    frame_counter = 0  # Initialize a frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only alternate frames
        if frame_counter % 1 == 0:
            # Process the frame for embeddings
            reg_img, embedding = registeration_form.get_embedding(frame)
            if embedding is not None:
                with open('face_embedding.txt', mode='ab') as f:
                    np.savetxt(f, embedding)

            # Convert the image to RGB and display in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, use_column_width=True)

        frame_counter += 1  # Increment the frame counter

    cap.release()


# Button to start/stop RTSP stream
if source == "RTSP Stream":
    if st.button("Start/Restart RTSP Stream"):
        show_rtsp_stream(rtsp_url)

# Submit button
if st.button('Submit'):
    return_value = registeration_form.save_data_in_redis_db(person_name, person_email)
    if return_value == True:
        st.success(f"{person_name} registered successfully")
    elif return_value == "name_false":
        st.error('Please enter your name: Name cannot be empty or spaces')
    elif return_value == 'file_false':
        st.error('face_embedding.txt is not found please refresh the page and execute again')



