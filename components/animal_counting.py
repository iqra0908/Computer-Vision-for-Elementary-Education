import logging
import threading
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from yolo_utils.helper import app_object_detection_yolov5

## Initiate logging
logger = logging.getLogger(__name__)

def animal_counting_UI():
    """
    The main UI function to display the Landing page UI
    """
    ## Set the page tab title
    st.title("Computer Vision for Elementary Education")
    st.subheader("Learning names of animals")

    ## Initiate the live video capture server with yolo detection
    app_object_detection_yolov5()

    ## Logging the live threads
    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")
    