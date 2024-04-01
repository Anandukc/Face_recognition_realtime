# code for person detection + face recognition
import cv2
from ultralytics import YOLO
# Import the FaceRecognizer class
from face_recognition import FaceRecognizer

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize the FaceRecognizer
face_recognizer = FaceRecognizer()

# Open the video file or camera
cap = cv2.VideoCapture(0)

# Loop through the video frames or camera feed
while cap.isOpened():
    # Read a frame
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame for person detection (class 0)
        results = model(frame, classes=[0], verbose=False, conf=0.45)
        boxes = results[0].boxes.xyxy.cpu()
        person_detected = len(boxes) > 0
        
        # If a person is detected, perform face recognition
        if person_detected:
            # Perform face recognition on the frame
            frame_with_recognition, recognized_faces = face_recognizer.recognize_faces(frame)
            
            # For each recognized face, you can process further as needed
            for person_name, bbox in recognized_faces:
                print(f"Recognized: {person_name}")
            
            # Display the frame with face recognition results
            cv2.imshow("YOLOv8 + Face Recognition", frame_with_recognition)
        else:
            # If no person is detected, just show the original frame
            cv2.imshow("YOLOv8 Inference", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End the loop if no more frames are available
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()


# #################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@######################################################
# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")
# names = model.model.names

# # Open the video file

# cap = cv2.VideoCapture(0)
# assert cap.isOpened(), "Error reading video file"

# while cap.isOpened():
#     success, im0 = cap.read()

#     if success:
#         results = model.predict(im0, show=False)

#         boxes = results[0].boxes.xyxy.cpu().tolist()
#         clss = results[0].boxes.cls.cpu().tolist()

#         annotator = Annotator(im0, line_width=3, example=names)

#         if boxes is not None:
#             for box, cls in zip(boxes, clss):
#                 if names[int(cls)] == "person":   # Check if person class is detected
#                     annotator.box_label(box, color=(255, 144, 31), label=names[int(cls)])
#                     print("Hello")  # Print "Hello" if person is detected

#         cv2.imshow("YOLOv8 Inference", annotator.im)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         continue

#     print("Video frame is empty or video processing has been successfully completed.")
#     break

# cap.release()
# cv2.destroyAllWindows()
# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# import cv2
# from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO('best.pt')

# results=model.predict(source="/home/anandu/insight_face_complete app/face_recognition_streamlit_demo/Selection-of-vegetables-02-99e06de.jpg", save=True)

# # Open the video file
# video_path = "/home/anandu/insight_face_complete app/face_recognition_streamlit_demo/production_id_4121625 (2160p).mp4"
# cap = cv2.VideoCapture(video_path)

# # Set the custom window size
# cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)  # This enables the window to be resizable
# cv2.resizeWindow("YOLOv8 Inference", 800, 600)  # Set your desired window size here

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()
