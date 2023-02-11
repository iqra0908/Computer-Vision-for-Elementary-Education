## Libraries imports
import cv2
import torch
from PIL import Image
import numpy as np

## Constants
Conf_threshold = 0.4
NMS_threshold = 0.4

def yolo_inference_on_image(image_name, im_size=224):
    """
    A function to run the yolo model over the input image and return the results
    with overlayed bounding boxes and labels

    Args:
    image_name -> input image to run inference on
    """

    ## Load custom trained Model using pytorch hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/yolov5.pt')
    
    ## Set IOU threshold and NMS threshold
    model.conf = Conf_threshold  # NMS confidence threshold
    model.iou = NMS_threshold  # NMS IoU threshold

    ## Read the image using openCV
    #image = cv2.imread(image_name)[..., ::-1]  # OpenCV image (BGR to RGB)
    ## Read the image using PIL
    image = Image.open(image_name)

    ## Run inference over the image
    results = model([image], size=im_size) 
    
    ## Get the bounding boxes and labels
    bbox_img = np.array(results.render()[0])

    ## Get the predictions dataframe
    predictions = results.pandas().xyxy[0]

    ## Save the image
    bbox_img = Image.fromarray(bbox_img)
    bbox_img.save("./data/results/predicted.jpg")

    ## Print the predictions
    print(predictions)

    parser.add_argument('-c', '--count') 

if __name__ == "__main__":
    import argparse

    ## Initialize the argument parser
    parser = argparse.ArgumentParser()

    ## Add the arguments
    parser.add_argument('--image', help='path to the image to run inference on', default="./data/images/test.jpg")
    parser.add_argument('--im_size', help='size of the image to be used for inference', default=224, type=int)
    
    ## Parse the arguments
    args = parser.parse_args()

    yolo_inference_on_image(args.image)
