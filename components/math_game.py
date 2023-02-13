## Library Imports
import numpy as np
import streamlit as st
from PIL import Image

## Local Imports
from media_pipe_utils.helper import math_test
from media_pipe_utils.helper import get_question

def counting_UI():
    """
    The main UI function to display the Math Game page UI
    """

    ## Set the page title
    st.title("Computer Vision for Elementary Education")
    st.subheader("Finger Math Game")

    problem_type = st.radio('Problem type', ('Addition', 'Subtraction', 'Mixed'))
    draw_hand_landmarks = True # not implementing yet

    st.write("Welcome to the Math Quiz using your fingers!")
    st.write("Answer the questions by raising the number of fingers corresponding to the answer")
    
    ## Get the question and answer
    if st.button('Generate Question'):
        question_string, answer = get_question(problem_type)
        st.write(question_string)
        st.session_state.problem_type = problem_type
        st.session_state.question = question_string
        st.session_state.answer = answer

    ## Get the image from the webcam
    img_file_buffer = st.camera_input("Answer with a picture!")

    ## Send the image to the check answer
    if img_file_buffer is not None:
        ## Read the image
        img = Image.open(img_file_buffer)
        ## To convert PIL Image to numpy array:
        img_array = np.array(img)

        ## Check the answer
        math_test(img_array, st.session_state.question, st.session_state.answer, st.session_state.problem_type, draw_landmarks = draw_hand_landmarks)
    else:
        st.write("No image yet!")
