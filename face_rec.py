import numpy as np
import pandas as pd
import cv2
import os
import redis

# import psycopg2
from datetime import date

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# time
import time
from datetime import datetime

# # connct to redis Client
# hostname = 'redis-13113.c275.us-east-1-4.ec2.cloud.redislabs.com'
# portnumber = 13113
# password = 'xl2Dawl8iLjj1tIIhZi27ab3CmC2JTyZ'

# r = redis.StrictRedis(host=hostname,
#                       port=portnumber,
#                       password=password)

hostname = 'localhost'
portnumber = 6379

r = redis.StrictRedis(host=hostname,
                      port=portnumber)


# retrive data from database
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_email', 'facial_features']
    retrive_df[['Name', 'email']] = retrive_df['name_email'].apply(lambda x: x.split('#')).apply(pd.Series)
    return retrive_df[['Name', 'email', 'facial_features']]


# configure face analysis

faceapp = FaceAnalysis(name='buffalo_l', root='insightface_model', providers=['CUDAExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=(0.5))


# ML search algorithm

def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_email=['Name', 'email'], thresh=0.5):  # cosine similarity

    dataframe = dataframe.copy()  ## step 1  tahe the dataframe  (colection of data)

    # step 2 index face embedding the df and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # step 3 cal . cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step 4 : filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step 5 : get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_email = data_filter.loc[argmax][name_email]

    else:
        person_name = 'Unknown'
        person_email = 'Unknown'

    return person_name, person_email


## real Time Prediction
# we need to save logs for every 1 mins
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], email=[], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], email=[], current_time=[])

    def saveLogs_redis(self):
        # Step 1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)

        # Step 2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name', inplace=True)

        # Step 3 : push data to Redis database  (list)
        # encode the data
        name_list = dataframe['name'].tolist()
        email_list = dataframe['email'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []

        for name, email, ctime in zip(name_list, email_list, ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}#{email}#{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('anandu_pred', *encoded_data)
        #  ### Anandu Code to push Data to AWS Cloud
        #     # Step 4: Push data to PostgreSQL RDS AWS database
        # self.push_data_to_aws_rds(name_list, email_list, ctime_list)

        self.reset_dict()

    # def push_data_to_aws_rds(self, name_list, email_list, ctime_list):
    #     # Database connection parameters

    #     # Database connection parameters
    #     db_params = {
    #         'dbname': 'lead_database',
    #         'user': 'postgres',
    #         'password': 'password',
    #         'host': 'localhost',
    #         'port': '5432'
    #     }

    #     # Connect to PostgreSQL database
    #     try:
    #         conn = psycopg2.connect(**db_params)
    #         cursor = conn.cursor()

    #         # Insert data into the table
    #         for name, email, ctime in zip(name_list, email_list, ctime_list):
    #             if name != 'Unknown':
    #                 ctime_date_only = ctime.split()[0]
    #                 insert_query = "INSERT INTO attendance_test (name, email, ctime) VALUES (%s, %s, %s);"
    #                 cursor.execute(insert_query, (name, email, ctime_date_only))
    #                 conn.commit()

    #         # Close cursor and connection
    #         cursor.close()
    #         conn.close()

    #     except Exception as e:
    #         print(f"Error encountered while pushing data to RDS: {e}")

    ##
    def face_prediction(self, test_image, dataframe, feature_column,
                        name_email=['Name', 'email'], thresh=0.5):
        # step 1 : find the time
        current_time = str(datetime.now())

        # step 1  : take the test image and apply to insight image
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        # step 2 : use for loop and extract the embedding and pass to ml_search algo

        identified_students = []
        ###
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                           feature_column,
                                                           test_vector=embeddings,
                                                           name_email=name_email,
                                                           thresh=thresh)
            identified_students.append(person_name)

            if person_name == 'Unknown':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color, 2)

            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
            # cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)

            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['email'].append(person_role)
            self.logs['current_time'].append(current_time)
        return test_copy, identified_students


#### Registeration Form

class RegisterationForm:
    def __init__(self):
        self.sample = 0

    def reset(self):
        self.sample = 0

    def get_embedding(self, frame):
        # get results from insightface model
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # put text samples
            text = f"samples = {self.sample}"
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings

    def save_data_in_redis_db(self, name, email):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}#{email}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        # if face_embedding.txt exits
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # step 1 : load 'face_embedding.txt'
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)  # flatten array

        # step 2 : convert into array   (proper shape)
        received_samples = int(x_array.size / 512)
        x_array = x_array.reshape(received_samples, 512)
        x_array = np.asarray(x_array)

        # step 3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # step 4 : save this into redis database
        # redis hashes
        # save key and facial bytes into radias database
        r.hset(name='embeddings', key=key, value=x_mean_bytes)

        # remove the file for each person
        os.remove('face_embedding.txt')
        self.reset()

        return True
