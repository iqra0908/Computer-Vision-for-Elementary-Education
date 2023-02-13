import streamlit as st
import mediapipe as mp
import random
from config import min_tracking_confidence, min_detection_confidence
from config import tipIds


def load_mediapipe(min_tracking_confidence, min_detection_confidence):
    """
    Load the mediapipe library modules for hand landmark detection

    Args:
        min_tracking_confidence (float): Minimum confidence value [0,1] for the hand tracking to be considered successful
        min_detection_confidence (float): Minimum confidence value [0,1] for the hand detection to be considered successful
    
    Returns:
        hands (object): mediapipe hands object
        mp_hands (object): mediapipe hands module
        mp_drawing (object): mediapipe drawing module
        mp_drawing_styles (object): mediapipe drawing styles module
    """
    # Mediapipe setup for hand landmark detection
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    # model settings
    hands = mp_hands.Hands(
        model_complexity = 0,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence
    )

    return hands, mp_hands, mp_drawing, mp_drawing_styles


def get_addition_prob():
    """
    Function to come up with random addition problem, making sure the answer is 10 or less

    Returns:
        question_string (str): The question string
        answer (int): The answer to the question
    """
    x = random.randint(1,9)
    y = random.randint(1,9)
    
    # Make sure the answer is 10 or less
    answer = max(x,y)
    addend1 = min(x,y)
    addend2 = answer-addend1
  
    assert addend1+addend2 == answer
    assert answer<11
    
    # Create the question string
    question_string = f'{addend1}+{addend2}='
  
    return question_string, answer


def get_subtraction_prob():
    """
    Function to come up with random subtraction problem, making sure the answer is 10 or less

    Returns:
        question_string (str): The question string
        answer (int): The answer to the question
    """

    x = random.randint(0,9)
    y = random.randint(0,9)
    
    # Make sure the answer is 10 or less
    value1 = max(x,y)
    value2 = min(x,y)
    answer = value1-value2
    assert answer>-1

    # Create the question string
    question_string = f'{value1}-{value2}='
  
    return question_string, answer


def get_question(which_type):
    """
    Function to get a question string and answer based on the type of question

    Args:
        which_type (str): The type of question to generate. Can be 'Addition', 'Subtraction', or 'Random'
    
    Returns:
        question_string (str): The question string
        answer (int): The answer to the question
    """

    if which_type == 'Addition':
        ## Get the addition problem
        question_string, answer = get_addition_prob()
    
    elif which_type == 'Subtraction':
        ## Get the subtraction problem
        question_string, answer = get_subtraction_prob()
  
    else:
        ## Get a random problem
        random_num = random.randint(0,10)
        if random_num<5:
            question_string, answer = get_addition_prob()
        else: 
            question_string, answer = get_subtraction_prob()
   
    return question_string, answer


def get_num_fingers(frame, hands):
    """
    Function to get the number of fingers raised in a frame

    Args:
        frame (np.array): The frame to process

    Returns:
        fingerCount (int): The number of fingers raised
        results.multi_hand_landmarks (list): The hand landmarks
    """

    ## Set count of fingers to 0 initially
    fingerCount = 0
    
    ## Process the frame
    results = hands.process(frame)

    ## If hand landmarks are detected
    if results.multi_hand_landmarks:

        ## Loop through each result
        for hand_landmarks in results.multi_hand_landmarks:    
            ## Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            ## Set variable to keep landmarks positions (x and y)
            handLandmarks = []

            ## Fill list with x and y positions of each landmark
            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])

            ## Test conditions for each finger: Count is increased if finger is raised
            ## Thumb: TIP x position must be greater or lower than IP x position, depending on hand label.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount = fingerCount+1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount = fingerCount+1

            for tip in tipIds:
                if handLandmarks[tip][1] < handLandmarks[tip-2][1]: #Index finger
                    fingerCount = fingerCount+1
                    
    return fingerCount, hand_landmarks


def math_test(image,  question_string, answer, problem_type, draw_landmarks = False):
    """
    Function to run the math test and display the results

    Args:
        image (np.array): The image to process
        question_string (str): The question string
        answer (int): The answer to the question
        draw_landmarks (bool): Whether to draw the hand landmarks on the image

    Returns:
        None
    """

    ## Load mediapipe
    hands, mp_hands, mp_drawing, mp_drawing_styles = load_mediapipe(min_tracking_confidence, min_detection_confidence)

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as hands:
        number_of_fingers, handlandmarks = get_num_fingers(frame=image, hands=hands)
        
        ## Display the question, answer and the user's answer
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("Question")
            #st.metric(label="Question", value=str(question_string[:-1]))
            st.info(problem_type + " " + question_string[:-1])

        with col2:    
            #st.metric(label="Right Answer", value=str(answer))
            st.write("Right Answer")
            st.warning(str(answer))
        with col3:
            st.write("Your Answer?")
            #st.metric(label="Your Answer", value=str(number_of_fingers))
            st.warning(str(number_of_fingers))
        with col4:
            st.write("Your Answer?")
            if number_of_fingers == answer:
                st.success("Correct", icon="✅")
                st.balloons()
            else:
                st.error("Incorrect", icon="🚨")

        ## Display the image and hand landmarks
        # st.image(image, use_column_width=True)
        # Draw the hand landmarks
        if draw_landmarks:
            mp_drawing.draw_landmarks(
                        image,
                        handlandmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        
        st.image(image, use_column_width=True)
