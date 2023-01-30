import mediapipe as mp 
import numpy as np
import cv2
from random import randrange
from finger_counter_webcam2 import get_finger_count


def get_addition_prob():
  x = randrange(10)
  y = randrange(10)
  answer = max(x,y)
  addend1 = min(x,y)
  addend2 = answer-addend1
  
  assert addend1+addend2 == answer
  assert answer<11
  
  return addend1, addend2, answer

num_correct = 0
num_probs = 5
for prob in range(num_probs):
  addend1, addend2, answer = get_addition_prob()
  print(f'{addend1}+{addend2}=')
  
  
  my_answer = get_finger_count()
  
  if my_answer == answer:
    print('Way to go! That is correct!')
    num_correct = num_correct + 1
    break
  elif np.abs(my_answer-answer) == 1:
    print('So close!')
  else:
    print('Incorrect.')
 
print(f'You got {num_correct} out of {num_probs} correct!')

  
 
