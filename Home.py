import streamlit as st
import pandas as pd
import numpy as np
import face_rec  # Assuming this is your module for face recognition and Redis operations
import redis

# Redis connection
hostname = 'localhost'
portnumber = 6379
r = redis.StrictRedis(host=hostname, port=portnumber)


# Function to retrieve and delete data
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


def delete_data(name_email_key):
    try:
        print(f"Attempting to delete key: {name_email_key}")  # Debug print
        response = r.hdel('embeddings', name_email_key)
        print(f"Deletion response: {response}")  # Debug print

        if response == 1:  # Redis hdel returns 1 if the key was deleted
            return True
        else:
            st.error(f"Could not delete {name_email_key}. Key not found.")
            return False
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False


# Streamlit app
def main():
    st.title('DalensAI')

    # Retrieve data from Redis
    with st.spinner('Retrieving Data from Redis DB ...'):
        redis_face_db = retrive_data(name='embeddings')
        st.success('Data successfully retrieved from Redis')

    # Search feature
    search_query = st.text_input("Search for a name")

    # Filter data based on search query
    if search_query:
        filtered_data = redis_face_db[
            redis_face_db.apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)]
    else:
        filtered_data = redis_face_db

    # Display data
    st.dataframe(filtered_data)

    # Create a list of labels for dropdown
    labels = [f"{row['Name']} ({row['email']})" for index, row in filtered_data.iterrows()]

    # Delete feature
    if labels:
        selected_label = st.selectbox("Select a name to delete", labels)
        selected_index = labels.index(selected_label)
        student_name, student_email = filtered_data.iloc[selected_index][['Name', 'email']]
        name_email_key = f"{student_name}#{student_email}"

        # Using a session state to keep track of the deletion confirmation
        if 'delete_confirmation' not in st.session_state:
            st.session_state['delete_confirmation'] = False

        if st.session_state['delete_confirmation']:
            if st.button(f"Confirm deletion of {student_name}?"):
                if delete_data(name_email_key):
                    st.success(f"{student_name} deleted successfully")
                    redis_face_db = retrive_data(name='embeddings')
                    st.dataframe(redis_face_db)
                    st.session_state['delete_confirmation'] = False
                else:
                    st.error("Deletion failed.")
        elif st.button("Delete Name"):
            st.session_state['delete_confirmation'] = True


if __name__ == "__main__":
    main()