import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model_with_batch_norm  # Assuming you have modified your modelutil.py

# Set page configuration
st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('https://149695847.v2.pressablecdn.com/wp-content/uploads/2020/03/liopa_header_video_bg-1.jpg')
    st.title('Lip Reading by Team-591865 (Soma Sekhar, Kowshik, Manoj)')
    st.info('This application is developed using the LipNet deep learning model with Batch Normalization.')

# File selection
options = os.listdir(os.path.join('C:\\Users\\krish\\Downloads\\lipReading\\data','s1'))
selected_video = st.selectbox('Choose video', options)

# Load data and model
video_path = os.path.join('C:\\Users\\krish\\Downloads\\lipReading\\data', 's1', selected_video)
data, annotations = load_data(tf.convert_to_tensor(video_path))
model = load_model_with_batch_norm()  # Use the model with Batch Normalization

# Display video and model predictions
col1, col2 = st.columns(2)

with col1:
    st.info('The video below displays the converted video in mp4 format')
    os.system(f'ffmpeg -i {video_path} -vcodec libx264 test_video.mp4 -y')
    video = open('C:\\Users\\krish\\Downloads\\lipReading\\app\\test_video.mp4', 'rb')
    video_bytes = video.read()
    st.video(video_bytes)

with col2:
    st.info('This is all the machine learning model sees when making a prediction')
    st.info('This is the output of the machine learning model as tokens')
    
    # Make predictions
    yhat = model.predict(tf.expand_dims(data, axis=0))
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    
    # Display decoder output
    st.text(decoder)

    st.info('Decode the raw tokens into words')
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    st.text(converted_prediction)

# Performance evaluation
st.header('Performance Evaluation with Batch Normalization')
with st.spinner('Evaluating performance...'):
    # Assuming you have a function for performance evaluation
    accuracy = evaluate_performance(model, data, annotations)
    st.success(f"Accuracy with Batch Normalization: {accuracy * 100:.2f}%")
