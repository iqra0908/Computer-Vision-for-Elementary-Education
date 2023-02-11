## Libraries to be imported
import av 
import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as np

## WebRTC to enable web servers to send and receive video streams over the network with low latency
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

## Create default WebRTC Client settings for browser
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)

## Local imports
from config import Conf_threshold, NMS_threshold
from config import model_config_file, model_weights
from config import class_names, COLORS

def load_yoloV5(model_path='./models/yolov5.pt'):
    """
    Loads the yoloV5 model from the given path
    """
    ## Load custom trained Model using pytorch hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    ## Set IOU threshold and NMS threshold
    model.conf = Conf_threshold  # NMS confidence threshold
    model.iou = NMS_threshold  # NMS IoU threshold

    return model

def yolov5_inference(image, model, im_size=640):
    """
    A function to run the yolo model over the input image and return the results
    with overlayed bounding boxes and labels

    Args:
    image -> input image to run inference on
    model -> yolo model object
    im_size -> size of the image to be used for inference

    returns:
    bbox_img -> image with bounding boxes and labels
    object_count -> number of objects detected
    predictions -> dataframe with predictions
    """

    ## Run inference over the image
    results = model([image], size=im_size) 
    
    ## Get the bounding boxes and labels
    bbox_img = np.array(results.render()[0])

    ## Get the number of objects detected
    object_count = len(results.pandas().xyxy[0])

    ## Get the predictions dataframe
    predictions = results.pandas().xyxy[0]

    return bbox_img, object_count, predictions

def load_yoloV3(model_weights, model_config_file, im_size=416):
    """ 
    Loads a yolo architecture from weights and config file and returns a model object
    which can be used for detection

    Args:
    model_weights -> path where the trained model weights are stored
    model_config_file -> path where current model file is stored

    Returns:
    model -> cv2.dnn_DetectionModel object with loaded pre trained yolo model
    """

    ## Initialize Yolo architecture
    net = cv2.dnn.readNet(model_weights, model_config_file)

    ## Select processing device
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ## Load pretrained model
    model = cv2.dnn_DetectionModel(net)

    ## Specify model inputs
    model.setInputParams(size=(im_size, im_size), scale=1/255, swapRB=True)

    return model


def app_object_detection():
    """
    The main function, which processes an input frame and run the default yolo V3 model over it
    to get object detection result. It further anotates the images with the bounding boxes
    of found objects
    """
    class VideoProcessorYolo(VideoProcessorBase):
        """
        A class that inherits VideoProcessorBase to process every frame from the live web 
        stream of camera. It takes a frame of the video stream as an input and applies object detection
        over it. After processing the results of object detection, we return the processed frame back
        """

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """
            A function to process incoming live frames from the video and return the processed frame 
            with object detection results

            Args:
            frame -> current live frame from camera stream

            Returns:
            processed_frame -> frame with overlayed results of object detection
            """

            ## Convert input frame to numpy array
            image = frame.to_ndarray(format="bgr24")

            ## Load pretrained trained yolo model
            model = load_yoloV3(model_weights, model_config_file)

            ## Fetch object detection result 
            classes, scores, boxes = model.detect(image, Conf_threshold, NMS_threshold)

            ## Loop through all detections
            for (classid, score, box) in zip(classes, scores, boxes):
                
                ## Select bounding box color
                color = COLORS[int(classid) % len(COLORS)]

                ## Draw a bouding box around the object
                cv2.rectangle(image, box, color, 1)

                ## Get the label for the predicted object
                label = "%s : %f" % (class_names[classid[0]], score)

                ## Add label next to the object
                cv2.putText(image, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
                
                ## Add count of object to the image
                cv2.putText(image, "Count: " + str(len(classes)), (10,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    ## Create a webrtc context to stream the video
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=VideoProcessorYolo,
        async_processing=True,
    )

def app_object_detection_yolov5(model_path='./models/yolov5.pt', im_size=640):
    """
    The main function, which processes an input frame and run the yolo custom V5 model over it
    to get object detection result. It further anotates the images with the bounding boxes
    of found objects
    """
    class VideoProcessorYolo(VideoProcessorBase):
        """
        A class that inherits VideoProcessorBase to process every frame from the live web 
        stream of camera. It takes a frame of the video stream as an input and applies object detection
        over it. After processing the results of object detection, we return the processed frame back
        """
        def __init__(self) -> None:
            super().__init__()
            ## Load pretrained trained yolo model
            self.model = load_yoloV5(model_path)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """
            A function to process incoming live frames from the video and return the processed frame 
            with object detection results from Yolo V5

            Args:
            frame -> current live frame from camera stream

            Returns:
            processed_frame -> frame with overlayed results of object detection
            """

            ## Convert input frame to numpy array
            image = frame.to_ndarray(format="bgr24")
            
            ## Flip the image and create a PIL image
            flipped = image[:, ::-1, :]
            im_pil = Image.fromarray(flipped)

            ## Fetch object detection result
            results = self.model([im_pil], size=im_size) 
            
            ## Get the bounding boxes and labels
            bbox_img = np.array(results.render()[0]) # updates results.imgs with boxes and labels
            
            ## Get the number of objects detected
            object_count = len(results.pandas().xyxy[0])

            ## Add count of object to the image
            #cv2.putText(bbox_img, "Count: " + str(object_count), (10,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 1, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")
    
    ## Create a webrtc context to stream the video
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=VideoProcessorYolo,
        async_processing=True,
    )