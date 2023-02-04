import mediapipe as mp 
import numpy as np
import cv2
from random import randrange
from finger_counter_webcam2 import get_finger_count

#values that could come from sliders/buttons in streamlit
num_probs = 5
which_type = 'subraction'
countdown = 10


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
  if which_type = 'addition':
    question_string, answer = get_addition_prob()
    
  elif which_type = 'subtraction':
    question_string, answer = get_subtraction_prob()
  
  else:
    random_num = randrange(10)
    if random_num<5:
      question_string, answer = get_addition_prob()
    else: 
      question_string, answer = get_subtraction_prob()
   
  #get the answer from the webcam module
  my_answer = get_finger_count(countdown = countdown, question_string = question_string)
  
  #compare given answer with correct answer
  if my_answer == answer:
    print('Way to go! That is correct!')
    num_correct = num_correct + 1

  elif np.abs(my_answer-answer) == 1:
     print('So close!')
  else:
    print('Incorrect.')
 
print(f'You got {num_correct} out of {num_probs} correct!')

  
