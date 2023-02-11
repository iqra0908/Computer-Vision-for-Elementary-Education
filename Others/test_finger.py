#### Finger counter using webcam to answer simple arithmetic problems
#### Heavily borrows from mediapipe website explaining mediapipe at https://google.github.io/mediapipe/solutions/hands.html
#### to link with streamlit, some values that could be sliders/buttons include:
####       num_probs = number of problems
####.      which_type = type of problems, including 'addition', 'subtraction' or 'mixed' (although anything not addition or subtraction = mixed
####       countdown = time in seconds to allow for answer of each problem

#### Note: still need exit key to stop the math test...ESC exits the webcam for each problem, but will still cycle through to num_probs



import mediapipe as mp 
import numpy as np
import cv2
from random import randrange
from others.finger_counter_webcam2 import get_finger_count

#values that could come from sliders/buttons in streamlit
num_probs = 5
which_type = 'addition'
countdown = 10

# Function to come up with random addition problem, making sure the answer is 10 or less
def get_addition_prob():
  x = randrange(10)
  y = randrange(10)
  answer = max(x,y)
  addend1 = min(x,y)
  addend2 = answer-addend1
  
  assert addend1+addend2 == answer
  assert answer<11
  
  question_string = f'{addend1}+{addend2}='
  
  return question_string, answer

# Function to come up with random subtraction proble making sure the answer is between 0 and 10
def get_subtraction_prob():
  x = randrange(10)
  y = randrange(10)
  value1 = max(x,y)
  value2 = min(x,y)
  answer = value1-value2
  question_string = f'{value1}-{value2}='
  
  assert answer>-1
  
  return question_string, answer

#start a group of questions
num_correct = 0

for prob in range(num_probs):
  
  #get the question
  if which_type == 'addition':
    question_string, answer = get_addition_prob()
    
  elif which_type == 'subtraction':
    question_string, answer = get_subtraction_prob()
  
  else:
    random_num = randrange(10)
    if random_num<5:
      question_string, answer = get_addition_prob()
    else: 
      question_string, answer = get_subtraction_prob()
   
  #get the answer from the webcam module, sends the question string to display on webcam
  my_answer = get_finger_count(countdown = countdown, question_string = question_string)
  
  #compare given answer with correct answer
  if my_answer == answer:
    print('Way to go! That is correct!')
    num_correct = num_correct + 1

  elif np.abs(my_answer-answer) == 1:
     print('So close!')
  else:
    print('Incorrect.')

# this and the feedback from each question could be output to streamlit not currently on the webcam display
print(f'You got {num_correct} out of {num_probs} correct!')

  
