import time
import cv2
import torch
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
import os

from utils import plot_boxes, plot_boxes_webcam
from darknet.darknet import Darknet
