# face recognition program
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import redis

class FaceRecognizer:
    def __init__(self, redis_host='localhost', redis_port=6379, hash_name='embeddings', providers=['CUDAExecutionProvider'], det_thresh=0.5):
        self.faceapp = FaceAnalysis(name='buffalo_l', providers=providers)
        self.faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=det_thresh)
        self.redis_conn = redis.StrictRedis(host=redis_host, port=redis_port)
        self.hash_name = hash_name
        self.combined_df = self.retrieve_data()

    def retrieve_data(self):
        data = self.redis_conn.hgetall(self.hash_name)
        df = pd.DataFrame(data.items(), columns=['name_email', 'facial_features'])
        df['facial_features'] = df['facial_features'].apply(lambda x: np.frombuffer(x, dtype=np.float32))
        df['name_email'] = df['name_email'].str.decode('utf-8')
        df[['Name', 'email']] = df['name_email'].str.split('#', expand=True)
        return df[['Name', 'email', 'facial_features']]

    @staticmethod
    def ml_search_algorithm(dataframe, feature_column, test_vector, name_email=['Name', 'email'], thresh=0.5):
        dataframe = dataframe.copy()
        X_list = dataframe[feature_column].tolist()
        x = np.asarray(X_list)
        similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
        similar_arr = np.array(similar).flatten()
        dataframe['cosine'] = similar_arr

        data_filter = dataframe.query(f'cosine >= {thresh}')
        if len(data_filter) > 0:
            data_filter.reset_index(drop=True, inplace=True)
            argmax = data_filter['cosine'].argmax()
            return data_filter.loc[argmax][name_email]
        else:
            return 'Unknown', 'Unknown'

    def recognize_faces(self, frame):
        results = self.faceapp.get(frame)
        recognized_faces = []
        for res in results:
            person_name, person_email = self.ml_search_algorithm(self.combined_df, 'facial_features', res['embedding'], thresh=0.45)
            bbox = res['bbox'].astype(int)
            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, person_name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            recognized_faces.append((person_name, bbox))
        return frame, recognized_faces

# Example usage
if __name__ == "__main__":
    face_recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_recognition, recognized_faces = face_recognizer.recognize_faces(frame)
        cv2.imshow('Real-time Face Recognition', frame_with_recognition)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

