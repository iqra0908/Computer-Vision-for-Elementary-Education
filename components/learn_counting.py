## Library Imports
import logging
import threading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

## Local Imports
from yolo_utils.helper import app_object_detection


## Initiate logging
logger = logging.getLogger(__name__)

def counting_UI():
    """
    The main UI function to display the Counting page UI
    """

    ## Set the page title
    st.title("Computer Vision for Elementary Education")
    st.subheader("Learning to Count")

    ## Initiate the live video capture server with yolo detection
    app_object_detection()

    ## Logging the live threads
    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")