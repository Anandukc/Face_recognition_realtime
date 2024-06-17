from Home import st
from Home import face_rec
import time
import cv2

st.subheader('Real-Time Attendance System')

# Retrieve data from Redis
with st.spinner('Retrieving Data from Redis DB ...'):
    redis_face_db = face_rec.retrive_data(name='embeddings')
    st.dataframe(redis_face_db)

st.success('Data successfully retrieved from Redis')

waitTime = 30  # time in seconds
realtimepred = face_rec.RealTimePred()  # real time prediction class

class_B_1 = 0       # Camera IP Address
cap_class_B_1 = cv2.VideoCapture(class_B_1)

# cv2.namedWindow('class_B_1 CCTV Footage with Face Recognition', cv2.WINDOW_NORMAL)

setTime = time.time()

frame_counter = 0  # Initialize a frame counter

while True:
    ret, frame = cap_class_B_1.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Process every other frame
    if frame_counter % 25 == 0:
        # Resize the frame to fit the screen resolution
        frame_resized = cv2.resize(frame, (640, 480))  # Adjust this to fit your screen

        img, students = realtimepred.face_prediction(frame_resized, redis_face_db, 'facial_features', ['Name', 'email'], thresh=0.5)

        cv2.imshow('Real Time Prediction', img)

    frame_counter += 1  # Increment the frame counter

    timenow = time.time()
    if timenow - setTime >= waitTime:
        realtimepred.saveLogs_redis()
        print('Save Data to redis database')
        setTime = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_class_B_1.release()
cv2.destroyAllWindows()











