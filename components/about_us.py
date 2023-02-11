## Library imports
import streamlit as st
from PIL import Image

def about_us_UI():
    """
    The main UI function to display the About US page UI.
    """

    st.write("""
        The goal of this project is to create modules that teach children some elementary education using computer vision.
        This allows kids to learn in a fun way using their toys.

        We are doing this project as a part of our core curriculam at Duke University for Masters in Artificial Intelligence (Course: AIPI 540: Deep Learning Applications)
        """)

    st.markdown("""---""")
    st.subheader("The Team")

    ## Displays the team members
    row_1_col1, row_1_col2, row_1_col3 = st.columns(3)
    with row_1_col1:
        image = Image.open('data/images/test.png')
        st.image(image, caption="Archit")
    with row_1_col2:
        image = Image.open('data/images/test.png')
        st.image(image, caption="Chad")
    with row_1_col3:
        image = Image.open('data/images/test.png')
        st.image(image, caption="Iqra")