import cv2
import mediapipe as mp
import numpy as np
import time

DRAW_LANDMARK = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
tipIds=[8,12,16,20]

#countdown refers to the number of seconds before 
def get_finger_count(countdown = 10, question_string = '?'):
    start_time = time.time()
    allowed_time = countdown
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while (cap.isOpened()&(countdown>0)):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
    
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
            # Initially set finger count to 0 for each cap
            fingerCount = 0

    

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
            #print(hand_landmarks)
            #print(f'Index finger tip coordinates: ('),
            #print(f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, ')
            #print(f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})')
        

    
                    # Get hand index to check label (left or right)
                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    # Set variable to keep landmarks positions (x and y)
                    handLandmarks = []

                    # Fill list with x and y positions of each landmark
                    for landmarks in hand_landmarks.landmark:
                        handLandmarks.append([landmarks.x, landmarks.y])

                    # Test conditions for each finger: Count is increased if finger is raised

                    # Thumb: TIP x position must be greater or lower than IP x position,depending on hand label.
                    if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                        fingerCount = fingerCount+1
                    elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                        fingerCount = fingerCount+1

        # Other fingers: TIP y position must be lower than PIP y position, 
        #   as image origin is in the upper left corner.
        # if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
        #   fingerCount = fingerCount+1
        # if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
        #   fingerCount = fingerCount+1
        # if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
        #   fingerCount = fingerCount+1
        # if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
        #   fingerCount = fingerCount+1

                    for tip in tipIds:
                        if handLandmarks[tip][1] < handLandmarks[tip-2][1]:       #Index finger
                            fingerCount = fingerCount+1

                    if DRAW_LANDMARK:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
    
    #  if len(lmList) !=0:
    #     fingers=[]

    #     # Thumb
    #     if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)

    #     for id in range(1,5):  #y axis
    #         if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)

    #     totalFingers=fingers.count(1)
    #     print(totalFingers)

    # Flip the image horizontally for a selfie-view display.
    #display finger count
    #cv2.flip(image,1)
    
    
            #cv2.imshow('MediaPipe Hands', image)
            image = cv2.flip(image, 1)
            image = cv2.putText(image, question_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            countdown = int(start_time+allowed_time - time.time())
            image = cv2.putText(image, f'{str(fingerCount)}', (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            countdown = int(start_time+allowed_time - time.time())
            image= cv2.putText(image, f'Time remaining: {countdown}', (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            cv2.imshow('MediaPipe Hands', image)
            #cv2.imshow('MediaPipe Hands', cv2.flip(image,1))
    
            if cv2.waitKey(5) & 0xFF == 27:
                #print(fingerCount)
                break
    time.sleep(1)
    cap.release()
    return fingerCount
            
    
