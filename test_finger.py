import mediapipe as mp 
import numpy as np
import cv2
from finger_counter_webcam2 import get_finger_count

my_answer = get_finger_count()
print(my_answer)