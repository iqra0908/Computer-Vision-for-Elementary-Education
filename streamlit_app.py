## Library imports
import pandas as pd 
import numpy as np 
import streamlit as st
from streamlit import runtime

## Local imports
from components import animal_counting
from components import about_us
from components import learn_counting
from components import animal_classification
from config import PAGES

## Landing page UI
def run_UI():
    """
    The main UI function to display the Landing page UI
    """
    ## Set the page tab title
    st.set_page_config(page_title="Elementary Education", page_icon="🤖")

    ## Set the page title and navigation bar
    st.sidebar.title('Learning Modules')
    if st.session_state["page"]:
        page=st.sidebar.radio('Navigation', PAGES, index=st.session_state["page"])
    else:
        page=st.sidebar.radio('Navigation', PAGES, index=0)
    st.experimental_set_query_params(page=page)


    ## Display the page selected on the navigation bar
    if page == 'About Us':
        st.sidebar.write("""
            ## About
            
            About Us
        """)
        st.title("About Us")
        about_us.about_us_UI()

    elif page == 'Learn Counting':
        st.sidebar.write("""
            ## About
            
            The goal of this project is to create modules that teach children some elementary education using computer vision.
            This allows kids to learn in a fun way using their toys.
        """)
        learn_counting.counting_UI()
    
    elif page == 'Learn Animals Names':
        st.sidebar.write("""
            ## About
            
            The goal of this project is to create modules that teach children some elementary education using computer vision.
            This allows kids to learn in a fun way using their toys.
        """)
        animal_classification.animal_classification_UI()

    else:
        st.sidebar.write("""
            ## About
            
            The goal of this project is to create modules that teach children some elementary education using computer vision.
            This allows kids to learn in a fun way using their toys.
        """)

        animal_counting.animal_counting_UI()


if __name__ == '__main__':
    ## Load the streamlit app with "Animal Classifier" as default page
    if runtime.exists():

        ## Get the page name from the URL
        url_params = st.experimental_get_query_params()

        if 'loaded' not in st.session_state:
            if len(url_params.keys()) == 0:
                ## Set the default page as "Animal Classifier"
                st.experimental_set_query_params(page='Learn Animals Names')
                url_params = st.experimental_get_query_params()
                
            ## Set the page index
            st.session_state.page = PAGES.index(url_params['page'][0])
        
        ## Call the main UI function
        run_UI()