## Library imports
import streamlit as st
from PIL import Image

## Local imports
from yolo_utils.helper import yolov5_inference, load_yoloV5
from resnet_utils.helper import predict


def animal_classification_UI():
    """
    The main UI function to display the page UI for animal classification
    """
    
    ## Page title
    st.title("Computer Vision for Elementary Education")
    st.subheader("Learning names of animals")

    ## Load the model
    model = load_yoloV5("./models/yolov5.pt", )

    ## Image uploader
    image_upload = st.file_uploader("Upload an image", type="jpg")

    if image_upload is not None:
        ## Load image
        image = Image.open(image_upload)

        ## Get predictions of the image
        result_img, object_count, predictions = yolov5_inference(image, model, im_size=416)

        ## Show the image with bounding boxes
        st.image(result_img, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        ## Show the detected objects
        count = 0
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        for i in range(len(predictions)):
            with columns[count]:
                st.metric(label=predictions["name"][i].upper(), value=format(predictions["confidence"][i]*100, "0.01f")+"%")
                count += 1
                if count >= 3:
                    count = 0
        
        ## Show the count of animals
        st.metric(label="Count of Animals", value=str(object_count))

        